import os
import json
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import timm


def safe_normalize(x, dim=1, eps=1e-12):
    if x.numel() == 0:
        return x
    denom = x.norm(dim=dim, keepdim=True).clamp_min(eps)
    x = x / denom
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def sanitize_tensor(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class ImageFileDataset(Dataset):
    def __init__(self, img_dir, file_names, labels=None, transform=None):
        self.img_dir = img_dir
        self.file_names = file_names
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.file_names[idx])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return img, idx
        return img, self.labels[idx], idx


class FeatureTensorDataset(Dataset):
    def __init__(self, features, labels=None, soft_labels=None):
        self.features = sanitize_tensor(features.float())
        self.labels = labels
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = -1 if self.labels is None else int(self.labels[idx])
        s = torch.empty(0) if self.soft_labels is None else self.soft_labels[idx]
        s = sanitize_tensor(s.float()) if torch.is_tensor(s) and s.numel() > 0 else s
        return x, y, s


class MetricAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_classes)

    def encode(self, x):
        z = self.ln(sanitize_tensor(x))
        z = self.fc1(z)
        z = F.gelu(z)
        z = self.drop(z)
        z = self.fc2(z)
        z = safe_normalize(z, dim=1)
        return sanitize_tensor(z)

    def forward(self, x):
        z = self.encode(x)
        logits = sanitize_tensor(self.classifier(z))
        return z, logits


def batch_hard_triplet_loss(embeddings, labels, margin=0.2):
    embeddings = safe_normalize(sanitize_tensor(embeddings), dim=1)
    if embeddings.shape[0] <= 1:
        return embeddings.new_tensor(0.0)

    sim = sanitize_tensor(embeddings @ embeddings.T)
    dist = sanitize_tensor((2.0 - 2.0 * sim).clamp_min(0.0))

    labels = labels.view(-1)
    same = labels[:, None] == labels[None, :]
    diff = ~same
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    same = same & (~eye)

    pos_dist = dist.masked_fill(~same, -1e6)
    neg_dist = dist.masked_fill(~diff, 1e6)

    hardest_pos = pos_dist.max(dim=1).values
    hardest_neg = neg_dist.min(dim=1).values
    valid = (hardest_pos > -1e5) & (hardest_neg < 1e5)
    if valid.sum() == 0:
        return embeddings.new_tensor(0.0)
    return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()


def soft_cross_entropy(logits, soft_targets):
    logits = sanitize_tensor(logits)
    soft_targets = sanitize_tensor(soft_targets)
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
    logp = F.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()


def supervised_contrastive_loss(features, labels, temperature=0.12):
    features = safe_normalize(sanitize_tensor(features), dim=1)
    device = features.device
    labels = labels.view(-1)
    n = features.shape[0]
    if n <= 1:
        return features.new_tensor(0.0)

    sim = sanitize_tensor((features @ features.T) / temperature)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(n, device=device, dtype=torch.bool)
    exp_sim = torch.exp(sim).masked_fill(eye, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_mask = (labels[:, None] == labels[None, :]) & (~eye)
    pos_count = pos_mask.sum(dim=1)

    if (pos_count > 0).sum() == 0:
        return features.new_tensor(0.0)

    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp_min(
        1
    )
    loss = -mean_log_prob_pos[pos_count > 0].mean()
    return sanitize_tensor(loss)


class ClassificationAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.idx_to_cat_id = None
        self.cat_id_to_idx = None
        self.num_classes = None
        self.feature_dim = None
        self.input_size = 224

        self.memory_features = None
        self.memory_posteriors = None
        self.memory_reliability = None
        self.memory_hubness_reliability = None
        self.base_memory_features = None

        self.metric_model = None
        self.metric_out_dim = None

        self.prototype_matrix = None
        self.multi_prototypes = None
        self.prototype_counts = None
        self.multi_proto_tau = 0.12

        self.proto_blend_weight = 0.72
        self.knn_k = 32
        self.knn_temperature = 0.05

        self.rerank_weight = 0.18
        self.rerank_query_k = 24
        self.rerank_mem_k = 20
        self.rerank_temperature = 0.05
        self.rerank_support_strength = None
        self.mem_neighbor_idx = None
        self.mem_neighbor_sim = None

        self.adaptive_temp_min = 0.90
        self.adaptive_temp_max = 1.25

        self.lda_weight = 0.16
        self.lda_mean = None
        self.lda_whiten_scale = None
        self.lda_proj = None
        self.lda_class_means = None
        self.lda_class_var = None
        self.lda_dim = None

        self.radial_weight = 0.08
        self.radial_class_centers = None
        self.radial_class_var = None

        self.label_memory_smooth_temp = 0.10
        self.label_memory_smooth_min = 0.04
        self.label_memory_smooth_max = 0.18

        self.unlabeled_memory_reliability_gamma = 1.35
        self.unlabeled_memory_reliability_floor = 0.20

        self.posterior_graph_k = 18
        self.posterior_graph_temperature = 0.06
        self.posterior_denoise_alpha = 0.82
        self.posterior_denoise_iters = 10
        self.posterior_labeled_clamp = 0.92
        self.posterior_denoise_blend = 0.35

        self.hubness_k = 24
        self.hubness_gamma = 1.10
        self.hubness_floor = 0.35

        self.adapt_smooth_k = 8
        self.adapt_smooth_self_weight = 0.72
        self.adapt_smooth_temperature = 0.06
        self.query_adapt_smooth_k = 10
        self.query_adapt_self_weight = 0.76
        self.query_adapt_smooth_temperature = 0.06

        self.memory_condense_enabled = True
        self.memory_condense_max_per_class = 36
        self.memory_condense_min_per_class = 10
        self.memory_condense_conf_thresh = 0.58
        self.memory_condense_sim_thresh = 0.985

        # New inference-only tweak: TTA-view feature consistency correction.
        self.view_consistency_enabled = True
        self.view_consensus_strength = 0.22
        self.view_low_agree_boost = 0.22
        self.view_memory_post_smooth_strength = 0.10

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _build_backbone(self):
        candidate_models = [
            "vit_base_patch14_dinov2.lvd142m",
            "vit_base_patch16_224.dino",
            "eva02_base_patch14_224.mim_in22k",
            "vit_base_patch16_clip_224.openai",
            "convnext_base.clip_laion2b",
            "resnet50.a1_in1k",
        ]
        model = None
        for name in candidate_models:
            try:
                model = timm.create_model(name, pretrained=True, num_classes=0)
                break
            except Exception:
                continue
        if model is None:
            model = timm.create_model("resnet50", pretrained=True, num_classes=0)

        model.to(self.device)
        for p in model.parameters():
            p.requires_grad = False

        cfg = getattr(model, "pretrained_cfg", None) or getattr(
            model, "default_cfg", {}
        )
        input_size = cfg.get("input_size", None)
        if input_size is not None and len(input_size) == 3:
            self.input_size = int(input_size[-1])

        self.model = model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_coco_train(self, ann_file):
        with open(ann_file, "r") as f:
            data = json.load(f)

        images = data["images"]
        annotations = data["annotations"]
        categories = data["categories"]

        self.idx_to_cat_id = [c["id"] for c in categories]
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.idx_to_cat_id)}
        self.num_classes = len(self.idx_to_cat_id)

        image_id_to_file = {img["id"]: img["file_name"] for img in images}
        file_names, labels = [], []
        for ann in annotations:
            file_names.append(image_id_to_file[ann["image_id"]])
            labels.append(self.cat_id_to_idx[ann["category_id"]])
        return file_names, labels

    def _load_coco_unlabeled(self, ann_file):
        with open(ann_file, "r") as f:
            data = json.load(f)
        return [img["file_name"] for img in data["images"]]

    def _bn_recalibrate(self, img_dir, train_files, unlabel_files):
        bn_modules = []
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                bn_modules.append(m)
        if len(bn_modules) == 0:
            return

        files = train_files + unlabel_files
        if len(files) == 0:
            return

        ds = ImageFileDataset(
            img_dir=img_dir, file_names=files, labels=None, transform=self.transform
        )
        loader = DataLoader(
            ds,
            batch_size=64,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=torch.cuda.is_available(),
        )

        self.model.eval()
        for m in bn_modules:
            m.train()

        with torch.no_grad():
            for batch in loader:
                imgs, _ = batch
                imgs = imgs.to(self.device, non_blocking=True)
                _ = self.model(imgs)

        self.model.eval()

    @torch.no_grad()
    def _extract_features(self, dataset, batch_size=64):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=torch.cuda.is_available(),
        )

        all_feats, all_labels, all_indices = [], [], []
        for batch in loader:
            if len(batch) == 3:
                images, labels, indices = batch
                all_labels.append(labels)
            else:
                images, indices = batch

            images = images.to(self.device, non_blocking=True)
            feats = self.model(images)
            feats = safe_normalize(sanitize_tensor(feats), dim=1)

            all_feats.append(feats.cpu())
            all_indices.append(indices)

        all_feats = torch.cat(all_feats, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        order = torch.argsort(all_indices)
        all_feats = sanitize_tensor(all_feats[order])

        if all_labels:
            all_labels = torch.cat(all_labels, dim=0)[order]
            return all_feats, all_labels
        return all_feats

    def _compute_prototypes(self, features, labels):
        features = sanitize_tensor(features)
        prototypes = []
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                proto = sanitize_tensor(features[mask].mean(dim=0))
            else:
                proto = torch.zeros(
                    features.shape[1], dtype=features.dtype, device=features.device
                )
            proto = safe_normalize(proto.unsqueeze(0), dim=1).squeeze(0)
            prototypes.append(proto)
        return sanitize_tensor(torch.stack(prototypes, dim=0))

    @torch.no_grad()
    def _smooth_features_knn(
        self, features, k=12, chunk_size=512, self_weight=0.5, temperature=0.07
    ):
        if features.shape[0] <= 1:
            return sanitize_tensor(features)

        feats_gpu = features.to(self.device, non_blocking=True)
        feats_gpu = safe_normalize(sanitize_tensor(feats_gpu), dim=1)
        n = feats_gpu.shape[0]
        out_chunks = []
        k_eff = min(k + 1, n)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            q = feats_gpu[start:end]
            sims = sanitize_tensor(q @ feats_gpu.T)
            vals, idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)

            if k_eff > 1:
                idx = idx[:, 1:]
                vals = vals[:, 1:]
                weights = torch.softmax(vals / temperature, dim=1)
                neigh = feats_gpu[idx]
                neigh_avg = (weights.unsqueeze(-1) * neigh).sum(dim=1)
            else:
                neigh_avg = q

            smoothed = self_weight * q + (1.0 - self_weight) * neigh_avg
            smoothed = safe_normalize(smoothed, dim=1)
            out_chunks.append(smoothed.cpu())

        return sanitize_tensor(torch.cat(out_chunks, dim=0))

    @torch.no_grad()
    def _query_smooth_against_memory(
        self, query_features, memory_features, k=10, self_weight=0.7, temperature=0.07
    ):
        query_features = safe_normalize(sanitize_tensor(query_features), dim=1)
        if memory_features is None or memory_features.shape[0] == 0:
            return query_features

        memory = safe_normalize(
            sanitize_tensor(memory_features.to(self.device, non_blocking=True)), dim=1
        )
        k_eff = min(k, memory.shape[0])
        if k_eff <= 0:
            return query_features

        sims = sanitize_tensor(query_features @ memory.T)
        vals, idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=False)
        weights = torch.softmax(vals / temperature, dim=1)

        if (
            self.memory_hubness_reliability is not None
            and memory.shape[0] == self.memory_hubness_reliability.shape[0]
        ):
            hub_rel = self.memory_hubness_reliability[idx]
            weights = weights * hub_rel
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        neigh = memory[idx]
        neigh_avg = (weights.unsqueeze(-1) * neigh).sum(dim=1)

        corrected = self_weight * query_features + (1.0 - self_weight) * neigh_avg
        corrected = safe_normalize(corrected, dim=1)
        return sanitize_tensor(corrected)

    @torch.no_grad()
    def _sinkhorn_balanced_assign(self, logits, epsilon=0.05, n_iters=30):
        logits = sanitize_tensor(logits)
        if logits.numel() == 0:
            return logits

        Q = torch.exp(logits / epsilon).T
        Q = sanitize_tensor(Q + 1e-12)
        Q = Q / Q.sum().clamp_min(1e-12)

        C, N = Q.shape
        r = torch.full((C,), 1.0 / C, device=Q.device, dtype=Q.dtype)
        c = torch.full((N,), 1.0 / N, device=Q.device, dtype=Q.dtype)

        for _ in range(n_iters):
            Q = Q * (r / (Q.sum(dim=1) + 1e-12)).unsqueeze(1)
            Q = Q * (c / (Q.sum(dim=0) + 1e-12)).unsqueeze(0)
            Q = sanitize_tensor(Q)

        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
        return sanitize_tensor(Q.T.contiguous())

    @torch.no_grad()
    def _refine_prototypes_with_sinkhorn(
        self,
        train_feats,
        train_labels,
        unlabel_feats,
        init_prototypes,
        unlabeled_weight=0.35,
        conf_power=2.0,
    ):
        train_feats = sanitize_tensor(train_feats)
        unlabel_feats = sanitize_tensor(unlabel_feats)
        init_prototypes = safe_normalize(sanitize_tensor(init_prototypes), dim=1)

        if unlabel_feats.shape[0] == 0:
            return init_prototypes, torch.empty(
                (0, self.num_classes), dtype=train_feats.dtype
            )

        device = self.device
        protos = init_prototypes.to(device)
        uf = safe_normalize(unlabel_feats.to(device), dim=1)
        tf = safe_normalize(train_feats.to(device), dim=1)
        tl = train_labels.to(device)

        sims = sanitize_tensor(uf @ protos.T)
        soft_q = self._sinkhorn_balanced_assign(sims, epsilon=0.05, n_iters=30)

        conf = soft_q.max(dim=1).values.pow(conf_power)
        weighted_q = soft_q * conf.unsqueeze(1)

        refined = []
        for c in range(self.num_classes):
            labeled_mask = tl == c
            labeled_feats_c = tf[labeled_mask]

            unl_w_c = weighted_q[:, c]
            if unl_w_c.sum() > 0:
                unl_proto_num = (unl_w_c.unsqueeze(1) * uf).sum(dim=0)
                unl_proto_den = unl_w_c.sum()
                unl_contrib = unl_proto_num / (unl_proto_den + 1e-12)
                if labeled_feats_c.shape[0] > 0:
                    proto = labeled_feats_c.mean(dim=0) + unlabeled_weight * unl_contrib
                else:
                    proto = unl_contrib
            else:
                proto = (
                    labeled_feats_c.mean(dim=0)
                    if labeled_feats_c.shape[0] > 0
                    else protos[c]
                )

            proto = safe_normalize(proto.unsqueeze(0), dim=1).squeeze(0)
            refined.append(proto)

        return sanitize_tensor(torch.stack(refined, dim=0)), sanitize_tensor(
            soft_q.cpu()
        )

    def _sample_beta(self, alpha=0.4):
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        lam = max(lam, 1.0 - lam)
        return float(lam)

    def _manifold_mixup_labeled(self, feats, labels):
        feats = sanitize_tensor(feats)
        if feats.shape[0] < 2:
            return None, None
        labels = labels.view(-1)
        classes = torch.unique(labels)
        if classes.numel() == 0:
            return None, None

        idx_a_list = []
        idx_b_list = []
        for c in classes.tolist():
            idx = torch.where(labels == c)[0]
            if idx.numel() < 2:
                continue
            perm = idx[torch.randperm(idx.numel(), device=idx.device)]
            pair_len = perm.numel() // 2
            if pair_len == 0:
                continue
            idx_a_list.append(perm[:pair_len])
            idx_b_list.append(perm[pair_len : pair_len * 2])

        if len(idx_a_list) == 0:
            return None, None

        idx_a = torch.cat(idx_a_list, dim=0)
        idx_b = torch.cat(idx_b_list, dim=0)
        n = min(idx_a.numel(), idx_b.numel())
        idx_a = idx_a[:n]
        idx_b = idx_b[:n]

        lam = self._sample_beta(alpha=0.4)
        mixed_x = lam * feats[idx_a] + (1.0 - lam) * feats[idx_b]
        mixed_y = F.one_hot(labels[idx_a], num_classes=self.num_classes).float()
        return sanitize_tensor(mixed_x), sanitize_tensor(mixed_y)

    def _manifold_mixup_unlabeled(self, feats, soft_targets):
        feats = sanitize_tensor(feats) if feats is not None else feats
        soft_targets = (
            sanitize_tensor(soft_targets) if soft_targets is not None else soft_targets
        )
        if feats is None or soft_targets is None or feats.shape[0] < 2:
            return None, None

        n = feats.shape[0]
        perm = torch.randperm(n, device=feats.device)
        pair_n = n // 2
        if pair_n == 0:
            return None, None

        idx_a = perm[:pair_n]
        idx_b = perm[pair_n : 2 * pair_n]
        pair_n = min(idx_a.numel(), idx_b.numel())
        idx_a = idx_a[:pair_n]
        idx_b = idx_b[:pair_n]

        lam = self._sample_beta(alpha=0.3)
        mixed_x = lam * feats[idx_a] + (1.0 - lam) * feats[idx_b]
        mixed_y = lam * soft_targets[idx_a] + (1.0 - lam) * soft_targets[idx_b]
        mixed_y = mixed_y / (mixed_y.sum(dim=1, keepdim=True) + 1e-12)
        return sanitize_tensor(mixed_x), sanitize_tensor(mixed_y)

    def _feature_augment_views(self, xb, noise_std=0.03, drop_prob=0.08):
        xb = sanitize_tensor(xb)
        noise1 = noise_std * torch.randn_like(xb)
        noise2 = noise_std * torch.randn_like(xb)
        mask1 = (torch.rand_like(xb) > drop_prob).float()
        mask2 = (torch.rand_like(xb) > drop_prob).float()
        v1 = safe_normalize((xb + noise1) * mask1, dim=1)
        v2 = safe_normalize((xb + noise2) * mask2, dim=1)
        return sanitize_tensor(v1), sanitize_tensor(v2)

    def _contrastive_polish_metric_adapter(self, model, train_feats, train_labels):
        train_feats = sanitize_tensor(train_feats)
        if train_feats.shape[0] <= 1:
            return

        ds = FeatureTensorDataset(train_feats, labels=train_labels, soft_labels=None)
        loader = DataLoader(
            ds,
            batch_size=min(256, max(96, len(ds))),
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
        model.train()
        polish_epochs = 10

        for _ in range(polish_epochs):
            for xb, yb, _ in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                v1, v2 = self._feature_augment_views(xb, noise_std=0.03, drop_prob=0.08)
                z1 = model.encode(v1)
                z2 = model.encode(v2)

                feats_cat = torch.cat([z1, z2], dim=0)
                labels_cat = torch.cat([yb, yb], dim=0)

                scl = supervised_contrastive_loss(
                    feats_cat, labels_cat, temperature=0.12
                )

                _, logits1 = model(v1)
                _, logits2 = model(v2)
                ce = 0.5 * (F.cross_entropy(logits1, yb) + F.cross_entropy(logits2, yb))

                loss = scl + 0.25 * ce

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()

    def _train_metric_adapter(
        self, train_feats, train_labels, unlabel_feats, unlabel_soft_q
    ):
        train_feats = sanitize_tensor(train_feats)
        unlabel_feats = sanitize_tensor(unlabel_feats)
        unlabel_soft_q = sanitize_tensor(unlabel_soft_q)

        in_dim = train_feats.shape[1]
        out_dim = min(384, in_dim)
        self.metric_out_dim = out_dim

        model = MetricAdapter(in_dim, out_dim, self.num_classes, dropout=0.1).to(
            self.device
        )

        pseudo_soft = None
        conf_mask = None
        if unlabel_feats.shape[0] > 0 and unlabel_soft_q.numel() > 0:
            conf = unlabel_soft_q.max(dim=1).values
            conf_mask = conf >= 0.55
            if conf_mask.any():
                pseudo_soft = unlabel_soft_q[conf_mask]

        lab_ds = FeatureTensorDataset(
            train_feats, labels=train_labels, soft_labels=None
        )
        lab_loader = DataLoader(
            lab_ds,
            batch_size=min(256, max(64, len(lab_ds))),
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        ul_loader = None
        if pseudo_soft is not None:
            ul_feats_kept = unlabel_feats[conf_mask]
            ul_ds = FeatureTensorDataset(
                ul_feats_kept, labels=None, soft_labels=pseudo_soft
            )
            ul_loader = DataLoader(
                ul_ds,
                batch_size=min(256, max(64, len(ul_ds))),
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        steps = 24
        ul_iter = iter(ul_loader) if ul_loader is not None else None

        model.train()
        for _ in range(steps):
            for xb, yb, _ in lab_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                z, logits = model(xb)
                ce = F.cross_entropy(logits, yb)
                tri = batch_hard_triplet_loss(z, yb, margin=0.2)
                loss = ce + 0.35 * tri

                mix_x_lab, mix_y_lab = self._manifold_mixup_labeled(xb, yb)
                if mix_x_lab is not None:
                    _, mix_logits_lab = model(mix_x_lab)
                    mix_loss_lab = soft_cross_entropy(
                        mix_logits_lab, mix_y_lab.to(self.device, non_blocking=True)
                    )
                    loss = loss + 0.18 * mix_loss_lab

                if ul_loader is not None:
                    try:
                        xub, _, sub = next(ul_iter)
                    except StopIteration:
                        ul_iter = iter(ul_loader)
                        xub, _, sub = next(ul_iter)

                    xub = xub.to(self.device, non_blocking=True)
                    sub = sub.to(self.device, non_blocking=True)

                    _, ulogits = model(xub)
                    soft_ce = soft_cross_entropy(ulogits, sub)
                    loss = loss + 0.25 * soft_ce

                    mix_x_ul, mix_y_ul = self._manifold_mixup_unlabeled(xub, sub)
                    if mix_x_ul is not None:
                        _, mix_logits_ul = model(mix_x_ul)
                        mix_loss_ul = soft_cross_entropy(mix_logits_ul, mix_y_ul)
                        loss = loss + 0.12 * mix_loss_ul

                loss = sanitize_tensor(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        model.eval()
        self._contrastive_polish_metric_adapter(model, train_feats, train_labels)
        self.metric_model = model

    @torch.no_grad()
    def _apply_metric_model(self, feats, batch_size=1024):
        feats = sanitize_tensor(feats)
        if self.metric_model is None:
            return feats

        outs = []
        for start in range(0, feats.shape[0], batch_size):
            x = feats[start : start + batch_size].to(self.device, non_blocking=True)
            z, _ = self.metric_model(x)
            outs.append(sanitize_tensor(z.cpu()))
        return sanitize_tensor(torch.cat(outs, dim=0))

    @torch.no_grad()
    def _build_multi_prototypes(
        self,
        train_feats_adapt,
        train_labels,
        unlabel_feats_adapt,
        unlabel_soft_q_adapt,
        base_prototypes,
        n_subproto=2,
        unl_conf_thresh=0.60,
        unlabeled_weight=0.30,
        kmeans_iters=4,
    ):
        train_feats_adapt = safe_normalize(sanitize_tensor(train_feats_adapt), dim=1)
        unlabel_feats_adapt = safe_normalize(
            sanitize_tensor(unlabel_feats_adapt), dim=1
        )
        unlabel_soft_q_adapt = sanitize_tensor(unlabel_soft_q_adapt)
        base_prototypes = safe_normalize(sanitize_tensor(base_prototypes), dim=1)

        protos = []
        counts = []

        if (
            unlabel_feats_adapt is not None
            and unlabel_feats_adapt.shape[0] > 0
            and unlabel_soft_q_adapt.numel() > 0
        ):
            unl_conf = unlabel_soft_q_adapt.max(dim=1).values
            unl_keep = unl_conf >= unl_conf_thresh
            uf = unlabel_feats_adapt[unl_keep]
            uq = unlabel_soft_q_adapt[unl_keep]
        else:
            uf = train_feats_adapt[:0]
            uq = torch.empty(
                (0, self.num_classes),
                dtype=train_feats_adapt.dtype,
                device=train_feats_adapt.device,
            )

        for c in range(self.num_classes):
            lf = train_feats_adapt[train_labels == c]
            class_base = base_prototypes[c].to(
                lf.device if lf.numel() > 0 else train_feats_adapt.device
            )

            if uf.shape[0] > 0:
                uw = uq[:, c]
                keep_u = uw > 1e-4
                uf_c = uf[keep_u]
                uw_c = (uw[keep_u] * unlabeled_weight).to(train_feats_adapt.device)
            else:
                uf_c = train_feats_adapt[:0]
                uw_c = train_feats_adapt.new_zeros((0,))

            if lf.shape[0] == 0 and uf_c.shape[0] == 0:
                class_protos = class_base.unsqueeze(0).repeat(n_subproto, 1)
                protos.append(safe_normalize(class_protos, dim=1).cpu())
                counts.append(1)
                continue

            labeled_weights = lf.new_ones((lf.shape[0],))
            feats_c = torch.cat([lf, uf_c], dim=0)
            weights_c = torch.cat(
                [labeled_weights, uw_c.to(labeled_weights.device)], dim=0
            )

            effective_k = 2 if feats_c.shape[0] >= 4 and lf.shape[0] >= 2 else 1

            if effective_k == 1:
                center = (weights_c.unsqueeze(1) * feats_c).sum(dim=0) / (
                    weights_c.sum() + 1e-12
                )
                center = safe_normalize(center.unsqueeze(0), dim=1).squeeze(0)
                class_protos = center.unsqueeze(0).repeat(n_subproto, 1)
                protos.append(sanitize_tensor(class_protos.cpu()))
                counts.append(1)
                continue

            sims_to_base = sanitize_tensor(
                feats_c.to(class_base.device)
                @ class_base.to(feats_c.device).unsqueeze(1)
            )
            order = torch.argsort(sims_to_base.squeeze(1))
            p0 = feats_c[order[0]]
            p1 = feats_c[order[-1]]
            centers = torch.stack([p0, p1], dim=0).to(self.device)
            centers = safe_normalize(centers, dim=1)

            x = feats_c.to(self.device)
            w = weights_c.to(self.device)

            for _ in range(kmeans_iters):
                sims = sanitize_tensor(x @ centers.T)
                assign = sims.argmax(dim=1)

                new_centers = []
                for j in range(2):
                    mask = assign == j
                    if mask.any():
                        ww = w[mask]
                        xx = x[mask]
                        cc = (ww.unsqueeze(1) * xx).sum(dim=0) / (ww.sum() + 1e-12)
                    else:
                        cc = centers[j]
                    cc = safe_normalize(cc.unsqueeze(0), dim=1).squeeze(0)
                    new_centers.append(cc)
                centers = sanitize_tensor(torch.stack(new_centers, dim=0))

            protos.append(centers.cpu())
            counts.append(2)

        self.multi_prototypes = sanitize_tensor(
            torch.stack(protos, dim=0).to(self.device)
        )
        self.prototype_counts = torch.tensor(
            counts, dtype=torch.long, device=self.device
        )
        self.prototype_matrix = sanitize_tensor(base_prototypes.to(self.device))

    @torch.no_grad()
    def _multi_proto_logits(self, features):
        features = safe_normalize(sanitize_tensor(features), dim=1)
        if self.multi_prototypes is None:
            return sanitize_tensor(features @ self.prototype_matrix.T)

        sims = torch.einsum("bd,ckd->bck", features, self.multi_prototypes)
        tau = self.multi_proto_tau
        logits = tau * torch.logsumexp(sims / tau, dim=2)

        if self.prototype_counts is not None:
            denom = self.prototype_counts.float().clamp_min(1.0).log().view(1, -1)
            logits = logits - tau * denom
        return sanitize_tensor(logits)

    @torch.no_grad()
    def _build_retrieval_memory(self):
        if self.memory_features is None or self.memory_posteriors is None:
            return

        mem = safe_normalize(sanitize_tensor(self.memory_features), dim=1)
        post = sanitize_tensor(self.memory_posteriors)
        n = mem.shape[0]
        if n == 0:
            return

        k_eff = min(self.rerank_mem_k + 1, n)
        sims = sanitize_tensor(mem @ mem.T)
        vals, idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
        if k_eff > 1:
            vals = vals[:, 1:]
            idx = idx[:, 1:]
        else:
            vals = vals[:, :0]
            idx = idx[:, :0]

        self.mem_neighbor_idx = idx.contiguous()
        self.mem_neighbor_sim = vals.contiguous()

        if idx.shape[1] == 0:
            self.rerank_support_strength = torch.ones(
                (n,), device=self.device, dtype=mem.dtype
            )
            return

        neigh_post = post[idx]
        center_post = post.unsqueeze(1)
        class_agreement = (neigh_post * center_post).sum(dim=2)
        neigh_w = torch.softmax(vals / self.rerank_temperature, dim=1)
        support = (neigh_w * class_agreement).sum(dim=1)
        support = 0.5 + 0.5 * support.clamp(0.0, 1.0)
        self.rerank_support_strength = sanitize_tensor(support.contiguous())

    @torch.no_grad()
    def _fit_shrinkage_lda(
        self,
        train_feats_adapt,
        train_labels,
        unlabel_feats_adapt=None,
        unlabel_soft_q_adapt=None,
        max_dim=32,
        unl_conf_thresh=0.85,
        unlabeled_mean_weight=0.10,
    ):
        x = sanitize_tensor(train_feats_adapt.float().to(self.device))
        y = train_labels.long().to(self.device)
        n, d = x.shape
        c = self.num_classes

        global_mean = sanitize_tensor(x.mean(dim=0))

        class_means = []
        class_counts = []
        for cls in range(c):
            mask = y == cls
            if mask.any():
                mu = sanitize_tensor(x[mask].mean(dim=0))
                cnt = int(mask.sum().item())
            else:
                mu = global_mean
                cnt = 0
            class_means.append(mu)
            class_counts.append(cnt)
        class_means = sanitize_tensor(torch.stack(class_means, dim=0))
        class_counts = torch.tensor(class_counts, device=self.device, dtype=x.dtype)

        if (
            unlabel_feats_adapt is not None
            and unlabel_soft_q_adapt is not None
            and unlabel_feats_adapt.shape[0] > 0
            and unlabel_soft_q_adapt.numel() > 0
        ):
            uf_cpu = sanitize_tensor(unlabel_feats_adapt.float().cpu())
            uq_cpu = sanitize_tensor(unlabel_soft_q_adapt.float().cpu())
            conf_cpu = uq_cpu.max(dim=1).values
            keep_cpu = conf_cpu >= unl_conf_thresh
            if keep_cpu.any():
                uf = uf_cpu[keep_cpu].to(self.device)
                uq = uq_cpu[keep_cpu].to(self.device)
                for cls in range(c):
                    w = uq[:, cls] * unlabeled_mean_weight
                    if w.sum() > 1e-8:
                        mu_u = (w.unsqueeze(1) * uf).sum(dim=0) / (w.sum() + 1e-12)
                        if class_counts[cls] > 0:
                            alpha = (
                                w.sum() / (class_counts[cls] + w.sum() + 1e-12)
                            ).clamp(0.0, 0.25)
                            class_means[cls] = safe_normalize(
                                (
                                    ((1.0 - alpha) * class_means[cls] + alpha * mu_u)
                                ).unsqueeze(0),
                                dim=1,
                            ).squeeze(0)
                        else:
                            class_means[cls] = safe_normalize(
                                mu_u.unsqueeze(0), dim=1
                            ).squeeze(0)

        centered = sanitize_tensor(x - global_mean)
        within_var = torch.zeros(d, device=self.device, dtype=x.dtype)
        valid_classes = 0
        for cls in range(c):
            mask = y == cls
            if mask.sum() >= 2:
                dif = sanitize_tensor(x[mask] - class_means[cls])
                within_var += dif.pow(2).mean(dim=0)
                valid_classes += 1
        if valid_classes > 0:
            within_var = within_var / valid_classes
        else:
            within_var = centered.pow(2).mean(dim=0)

        global_var = centered.pow(2).mean(dim=0)
        shrink = 0.35
        within_var = (1.0 - shrink) * within_var + shrink * global_var
        within_var = within_var.clamp_min(1e-4)
        whiten_scale = within_var.rsqrt()

        cmw = (class_means - global_mean) * whiten_scale.unsqueeze(0)
        weights = torch.sqrt(class_counts.clamp_min(1.0)).unsqueeze(1)
        B = sanitize_tensor(cmw * weights)

        rank = max(1, min(max_dim, c - 1, d))
        try:
            _, _, V = torch.pca_lowrank(B, q=rank, center=False)
            proj = V[:, :rank]
        except Exception:
            cov_b = sanitize_tensor(B.T @ B)
            evals, evecs = torch.linalg.eigh(cov_b)
            order = torch.argsort(evals, descending=True)
            proj = evecs[:, order[:rank]]

        z_train = ((x - global_mean) * whiten_scale.unsqueeze(0)) @ proj
        z_train = safe_normalize(z_train, dim=1)

        z_means = []
        z_var = []
        global_z_var = z_train.var(dim=0, unbiased=False).mean().clamp_min(1e-4)

        for cls in range(c):
            mask = y == cls
            if mask.any():
                zc = z_train[mask]
                mu = sanitize_tensor(zc.mean(dim=0))
                if zc.shape[0] >= 2:
                    rv = (zc - mu).pow(2).sum(dim=1).mean()
                else:
                    rv = global_z_var
            else:
                mu = torch.zeros(rank, device=self.device, dtype=z_train.dtype)
                rv = global_z_var
            mu = (
                safe_normalize(mu.unsqueeze(0), dim=1).squeeze(0)
                if mu.abs().sum() > 0
                else mu
            )
            z_means.append(mu)
            z_var.append(rv)

        z_means = sanitize_tensor(torch.stack(z_means, dim=0))
        z_var = sanitize_tensor(torch.stack(z_var, dim=0))
        z_var = 0.7 * z_var + 0.3 * z_var.mean()
        z_var = z_var.clamp_min(0.05)

        self.lda_mean = sanitize_tensor(global_mean)
        self.lda_whiten_scale = sanitize_tensor(whiten_scale)
        self.lda_proj = sanitize_tensor(proj)
        self.lda_class_means = sanitize_tensor(z_means)
        self.lda_class_var = sanitize_tensor(z_var)
        self.lda_dim = rank

    @torch.no_grad()
    def _lda_logits(self, features):
        features = sanitize_tensor(features)
        if (
            self.lda_mean is None
            or self.lda_whiten_scale is None
            or self.lda_proj is None
            or self.lda_class_means is None
            or self.lda_class_var is None
        ):
            return torch.zeros(
                (features.shape[0], self.num_classes),
                device=features.device,
                dtype=features.dtype,
            )

        z = (
            (features - self.lda_mean) * self.lda_whiten_scale.unsqueeze(0)
        ) @ self.lda_proj
        z = safe_normalize(z, dim=1)
        dif = z.unsqueeze(1) - self.lda_class_means.unsqueeze(0)
        dist2 = dif.pow(2).sum(dim=2)
        logits = -dist2 / self.lda_class_var.unsqueeze(0).clamp_min(1e-12)
        return sanitize_tensor(logits)

    @torch.no_grad()
    def _fit_radial_support_model(self, train_feats_adapt, train_labels):
        x = safe_normalize(
            sanitize_tensor(train_feats_adapt.to(self.device).float()), dim=1
        )
        y = train_labels.long().to(self.device)

        centers = []
        class_var = []
        global_center = sanitize_tensor(x.mean(dim=0))
        global_center = safe_normalize(global_center.unsqueeze(0), dim=1).squeeze(0)

        all_dist = []
        for c in range(self.num_classes):
            mask = y == c
            if mask.any():
                xc = x[mask]
                mu = sanitize_tensor(xc.mean(dim=0))
                mu = safe_normalize(mu.unsqueeze(0), dim=1).squeeze(0)
                dist2 = (xc - mu.unsqueeze(0)).pow(2).sum(dim=1)
                all_dist.append(dist2)
            else:
                mu = global_center
            centers.append(mu)

        if len(all_dist) > 0:
            global_var = torch.cat(all_dist, dim=0).mean().clamp_min(0.05)
        else:
            global_var = x.new_tensor(1.0)

        counts = torch.bincount(y, minlength=self.num_classes).float()

        for c in range(self.num_classes):
            mask = y == c
            if mask.any():
                xc = x[mask]
                mu = centers[c]
                dist2 = (xc - mu.unsqueeze(0)).pow(2).sum(dim=1)
                var_c = dist2.mean() if dist2.numel() > 0 else global_var
            else:
                var_c = global_var

            n_c = counts[c].item()
            shrink = 3.0 / (n_c + 3.0) if n_c > 0 else 1.0
            var_c = (1.0 - shrink) * var_c + shrink * global_var
            class_var.append(var_c.clamp_min(0.05))

        self.radial_class_centers = sanitize_tensor(torch.stack(centers, dim=0))
        self.radial_class_var = sanitize_tensor(torch.stack(class_var, dim=0))

    @torch.no_grad()
    def _radial_support_logits(self, features):
        features = safe_normalize(sanitize_tensor(features), dim=1)
        if self.radial_class_centers is None or self.radial_class_var is None:
            return torch.zeros(
                (features.shape[0], self.num_classes),
                device=features.device,
                dtype=features.dtype,
            )

        centers = safe_normalize(sanitize_tensor(self.radial_class_centers), dim=1)
        var = sanitize_tensor(self.radial_class_var).clamp_min(1e-12)
        dif = features.unsqueeze(1) - centers.unsqueeze(0)
        dist2 = dif.pow(2).sum(dim=2)
        logits = -dist2 / var.unsqueeze(0)
        return sanitize_tensor(logits)

    @torch.no_grad()
    def _build_smoothed_labeled_memory_posteriors(
        self, train_feats_adapt, train_labels, prototypes
    ):
        work_device = self.device
        feats = safe_normalize(
            sanitize_tensor(
                train_feats_adapt.to(work_device, non_blocking=True).float()
            ),
            dim=1,
        )
        labels = train_labels.long().to(work_device, non_blocking=True)
        protos = safe_normalize(
            sanitize_tensor(prototypes.to(work_device, non_blocking=True).float()),
            dim=1,
        )

        counts = torch.bincount(labels, minlength=self.num_classes).float()
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

        sims = sanitize_tensor(feats @ protos.T)
        soft = torch.softmax(sims / self.label_memory_smooth_temp, dim=1)

        class_smooth = counts[labels].clamp_min(1.0).rsqrt() * 0.42
        class_smooth = class_smooth.clamp(
            self.label_memory_smooth_min, self.label_memory_smooth_max
        )
        class_smooth = class_smooth.unsqueeze(1)

        post = (1.0 - class_smooth) * one_hot + class_smooth * soft

        true_idx = labels.view(-1, 1)
        true_mass = post.gather(1, true_idx)
        min_true = 0.78
        need_fix = true_mass.squeeze(1) < min_true
        if need_fix.any():
            post_fix = post[need_fix]
            idx_fix = true_idx[need_fix]
            non_true_scale = (1.0 - min_true) / (
                1.0 - post_fix.gather(1, idx_fix) + 1e-12
            )
            post_fix = post_fix * non_true_scale
            post_fix.scatter_(1, idx_fix, torch.full_like(idx_fix.float(), min_true))
            post[need_fix] = post_fix

        post = post / (post.sum(dim=1, keepdim=True) + 1e-12)
        return sanitize_tensor(post.cpu())

    @torch.no_grad()
    def _build_memory_reliability(self, n_labeled, memory_posteriors):
        memory_posteriors = sanitize_tensor(memory_posteriors)
        n_total = memory_posteriors.shape[0]
        rel = torch.ones(n_total, dtype=memory_posteriors.dtype)

        if n_total <= n_labeled:
            return rel

        unl_post = memory_posteriors[n_labeled:].float()
        entropy = -(unl_post * (unl_post.clamp_min(1e-12).log())).sum(dim=1)
        entropy = entropy / math.log(float(self.num_classes))
        reliability = (
            (1.0 - entropy).clamp(0.0, 1.0).pow(self.unlabeled_memory_reliability_gamma)
        )
        reliability = reliability.clamp_min(self.unlabeled_memory_reliability_floor)
        rel[n_labeled:] = reliability
        return sanitize_tensor(rel)

    @torch.no_grad()
    def _graph_denoise_memory_posteriors(
        self, memory_features, memory_posteriors, n_labeled
    ):
        if memory_features is None or memory_posteriors is None:
            return memory_posteriors
        n = memory_features.shape[0]
        if n <= 1:
            return sanitize_tensor(memory_posteriors)

        feats = safe_normalize(
            memory_features.to(self.device, non_blocking=True).float(), dim=1
        )
        post0 = sanitize_tensor(
            memory_posteriors.to(self.device, non_blocking=True).float()
        )
        k_eff = min(self.posterior_graph_k + 1, n)
        if k_eff <= 1:
            return sanitize_tensor(memory_posteriors)

        sims = sanitize_tensor(feats @ feats.T)
        vals, idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
        vals = vals[:, 1:]
        idx = idx[:, 1:]

        weights = torch.softmax(vals / self.posterior_graph_temperature, dim=1)

        Y = post0.clone()
        labeled_targets = post0[:n_labeled].clone()
        if n_labeled > 0:
            hard_labels = labeled_targets.argmax(dim=1)
            one_hot = F.one_hot(hard_labels, num_classes=self.num_classes).float()
            labeled_targets = (
                self.posterior_labeled_clamp * one_hot
                + (1.0 - self.posterior_labeled_clamp) * labeled_targets
            )
            labeled_targets = labeled_targets / (
                labeled_targets.sum(dim=1, keepdim=True) + 1e-12
            )

        for _ in range(self.posterior_denoise_iters):
            neigh_post = Y[idx]
            propagated = (weights.unsqueeze(-1) * neigh_post).sum(dim=1)
            Y = (
                self.posterior_denoise_alpha * propagated
                + (1.0 - self.posterior_denoise_alpha) * post0
            )
            if n_labeled > 0:
                Y[:n_labeled] = labeled_targets
            Y = Y.clamp_min(1e-12)
            Y = Y / (Y.sum(dim=1, keepdim=True) + 1e-12)
            Y = sanitize_tensor(Y)

        out = post0.clone()
        if n_labeled < n:
            out[n_labeled:] = (1.0 - self.posterior_denoise_blend) * post0[
                n_labeled:
            ] + self.posterior_denoise_blend * Y[n_labeled:]
            out[n_labeled:] = out[n_labeled:] / (
                out[n_labeled:].sum(dim=1, keepdim=True) + 1e-12
            )
        return sanitize_tensor(out.cpu())

    @torch.no_grad()
    def _build_hubness_reliability(self):
        if self.memory_features is None or self.memory_features.shape[0] <= 1:
            self.memory_hubness_reliability = None
            return

        mem = safe_normalize(sanitize_tensor(self.memory_features), dim=1)
        n = mem.shape[0]
        k_eff = min(self.hubness_k + 1, n)
        if k_eff <= 1:
            self.memory_hubness_reliability = torch.ones(
                n, device=self.device, dtype=mem.dtype
            )
            return

        sims = sanitize_tensor(mem @ mem.T)
        _, idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=False)
        idx = idx[:, 1:]

        counts = torch.bincount(idx.reshape(-1), minlength=n).float()
        expected = float(idx.numel()) / float(n)
        hubness = counts / max(expected, 1e-12)
        hubness = hubness / hubness.mean().clamp_min(1e-12)

        reliability = hubness.pow(-self.hubness_gamma)
        reliability = reliability / reliability.max().clamp_min(1e-12)
        reliability = reliability.clamp_min(self.hubness_floor)

        self.memory_hubness_reliability = sanitize_tensor(
            reliability.to(self.device).contiguous()
        )

    @torch.no_grad()
    def _condense_memory_bank(self, n_labeled):
        if not self.memory_condense_enabled:
            return
        if self.memory_features is None or self.memory_posteriors is None:
            return
        n_total = self.memory_features.shape[0]
        if n_total <= n_labeled or (n_total - n_labeled) < 64:
            return

        feats = safe_normalize(sanitize_tensor(self.memory_features), dim=1)
        post = sanitize_tensor(self.memory_posteriors)
        rel = self.memory_reliability
        if rel is None:
            rel = torch.ones(n_total, device=feats.device, dtype=feats.dtype)
        else:
            rel = sanitize_tensor(rel.to(feats.device))

        labeled_feats = feats[:n_labeled]
        labeled_post = post[:n_labeled]
        labeled_rel = rel[:n_labeled]

        unl_feats = feats[n_labeled:]
        unl_post = post[n_labeled:]
        unl_rel = rel[n_labeled:]

        if unl_feats.shape[0] == 0:
            return

        pred = unl_post.argmax(dim=1)
        conf = unl_post.max(dim=1).values
        proto = self.prototype_matrix
        if proto is None:
            return
        proto = safe_normalize(sanitize_tensor(proto.to(feats.device)), dim=1)

        sim_to_proto = (unl_feats * proto[pred]).sum(dim=1)
        quality = conf * unl_rel * (0.5 + 0.5 * sim_to_proto.clamp(0.0, 1.0))

        selected_global = []

        for c in range(self.num_classes):
            cls_mask = pred == c
            if not cls_mask.any():
                continue

            cls_idx = torch.where(cls_mask)[0]
            cls_conf = conf[cls_idx]
            cls_quality = quality[cls_idx]

            strong_mask = cls_conf >= self.memory_condense_conf_thresh
            if strong_mask.any():
                cls_idx = cls_idx[strong_mask]
                cls_quality = cls_quality[strong_mask]

            if cls_idx.numel() == 0:
                continue

            order = torch.argsort(cls_quality, descending=True)
            cls_idx = cls_idx[order]

            max_keep = min(self.memory_condense_max_per_class, cls_idx.numel())
            keep = []
            kept_feats = []

            for local_i in cls_idx.tolist():
                f = unl_feats[local_i]
                if len(keep) == 0:
                    keep.append(local_i)
                    kept_feats.append(f)
                else:
                    bank = torch.stack(kept_feats, dim=0)
                    mx = torch.max(bank @ f)
                    if (
                        mx.item() < self.memory_condense_sim_thresh
                        or len(keep) < self.memory_condense_min_per_class
                    ):
                        keep.append(local_i)
                        kept_feats.append(f)
                if len(keep) >= max_keep:
                    break

            if len(keep) > 0:
                selected_global.append(
                    torch.tensor(keep, device=feats.device, dtype=torch.long)
                )

        if len(selected_global) == 0:
            return

        selected_unl = torch.cat(selected_global, dim=0)
        selected_unl = torch.unique(selected_unl, sorted=True)

        new_feats = torch.cat([labeled_feats, unl_feats[selected_unl]], dim=0)
        new_post = torch.cat([labeled_post, unl_post[selected_unl]], dim=0)
        new_rel = torch.cat([labeled_rel, unl_rel[selected_unl]], dim=0)

        self.memory_features = sanitize_tensor(new_feats.contiguous())
        self.memory_posteriors = sanitize_tensor(new_post.contiguous())
        self.memory_reliability = sanitize_tensor(new_rel.contiguous())

    def fit(self, img_dir, train_ann_file, unlabel_ann_file):
        self._build_backbone()

        train_files, train_labels = self._load_coco_train(train_ann_file)
        unlabel_files = self._load_coco_unlabeled(unlabel_ann_file)

        self._bn_recalibrate(img_dir, train_files, unlabel_files)

        train_ds = ImageFileDataset(
            img_dir=img_dir,
            file_names=train_files,
            labels=train_labels,
            transform=self.transform,
        )
        unlabel_ds = ImageFileDataset(
            img_dir=img_dir,
            file_names=unlabel_files,
            labels=None,
            transform=self.transform,
        )

        train_feats, train_labels_tensor = self._extract_features(
            train_ds, batch_size=64
        )
        unlabel_feats = self._extract_features(unlabel_ds, batch_size=64)

        self.feature_dim = train_feats.shape[1]

        all_feats = (
            torch.cat([train_feats, unlabel_feats], dim=0)
            if len(unlabel_feats) > 0
            else train_feats
        )
        all_feats = self._smooth_features_knn(
            all_feats, k=12, chunk_size=512, self_weight=0.5, temperature=0.07
        )

        n_train = train_feats.shape[0]
        train_feats = sanitize_tensor(all_feats[:n_train])
        unlabel_feats = sanitize_tensor(
            all_feats[n_train:] if len(unlabel_feats) > 0 else all_feats[:0]
        )

        self.base_memory_features = sanitize_tensor(
            all_feats.to(self.device).contiguous()
        )

        init_prototypes = self._compute_prototypes(train_feats, train_labels_tensor)
        _, unlabeled_soft_q = self._refine_prototypes_with_sinkhorn(
            train_feats=train_feats,
            train_labels=train_labels_tensor,
            unlabel_feats=unlabel_feats,
            init_prototypes=init_prototypes,
            unlabeled_weight=0.35,
            conf_power=2.0,
        )

        self._train_metric_adapter(
            train_feats=train_feats,
            train_labels=train_labels_tensor,
            unlabel_feats=unlabel_feats,
            unlabel_soft_q=unlabeled_soft_q,
        )

        train_feats_adapt = sanitize_tensor(self._apply_metric_model(train_feats))
        unlabel_feats_adapt = (
            sanitize_tensor(self._apply_metric_model(unlabel_feats))
            if unlabel_feats.shape[0] > 0
            else unlabel_feats
        )

        all_feats_adapt = (
            torch.cat([train_feats_adapt, unlabel_feats_adapt], dim=0)
            if unlabel_feats_adapt.shape[0] > 0
            else train_feats_adapt
        )
        all_feats_adapt = self._smooth_features_knn(
            all_feats_adapt,
            k=self.adapt_smooth_k,
            chunk_size=512,
            self_weight=self.adapt_smooth_self_weight,
            temperature=self.adapt_smooth_temperature,
        )
        train_feats_adapt = sanitize_tensor(all_feats_adapt[:n_train])
        unlabel_feats_adapt = sanitize_tensor(
            all_feats_adapt[n_train:]
            if unlabel_feats_adapt.shape[0] > 0
            else all_feats_adapt[:0]
        )

        prototypes = self._compute_prototypes(train_feats_adapt, train_labels_tensor)
        prototypes, unlabeled_soft_q_adapt = self._refine_prototypes_with_sinkhorn(
            train_feats=train_feats_adapt,
            train_labels=train_labels_tensor,
            unlabel_feats=unlabel_feats_adapt,
            init_prototypes=prototypes,
            unlabeled_weight=0.35,
            conf_power=2.0,
        )

        self._build_multi_prototypes(
            train_feats_adapt=train_feats_adapt,
            train_labels=train_labels_tensor,
            unlabel_feats_adapt=unlabel_feats_adapt,
            unlabel_soft_q_adapt=unlabeled_soft_q_adapt,
            base_prototypes=prototypes,
            n_subproto=2,
            unl_conf_thresh=0.60,
            unlabeled_weight=0.30,
            kmeans_iters=4,
        )

        all_feats_adapt = (
            torch.cat([train_feats_adapt, unlabel_feats_adapt], dim=0)
            if unlabel_feats_adapt.shape[0] > 0
            else train_feats_adapt
        )
        self.memory_features = sanitize_tensor(
            all_feats_adapt.to(self.device).contiguous()
        )

        labeled_post = self._build_smoothed_labeled_memory_posteriors(
            train_feats_adapt=train_feats_adapt,
            train_labels=train_labels_tensor,
            prototypes=prototypes,
        )

        if unlabeled_soft_q_adapt.numel() > 0:
            memory_post = torch.cat(
                [labeled_post, sanitize_tensor(unlabeled_soft_q_adapt.float())], dim=0
            )
        else:
            memory_post = labeled_post

        memory_post = self._graph_denoise_memory_posteriors(
            memory_features=self.memory_features,
            memory_posteriors=memory_post,
            n_labeled=n_train,
        )

        self.memory_posteriors = sanitize_tensor(
            memory_post.to(self.device).contiguous()
        )

        mem_rel = self._build_memory_reliability(
            n_labeled=n_train, memory_posteriors=memory_post
        )
        self.memory_reliability = sanitize_tensor(mem_rel.to(self.device).contiguous())

        self._build_hubness_reliability()
        if self.memory_hubness_reliability is not None:
            self.memory_reliability = sanitize_tensor(
                (self.memory_reliability * self.memory_hubness_reliability)
                .clamp_min(0.10)
                .contiguous()
            )

        self._condense_memory_bank(n_labeled=n_train)

        self._build_hubness_reliability()
        if (
            self.memory_hubness_reliability is not None
            and self.memory_reliability is not None
        ):
            self.memory_reliability = sanitize_tensor(
                self.memory_reliability.clamp_min(0.10).contiguous()
            )

        self._build_retrieval_memory()

        self._fit_shrinkage_lda(
            train_feats_adapt=train_feats_adapt,
            train_labels=train_labels_tensor,
            unlabel_feats_adapt=unlabel_feats_adapt,
            unlabel_soft_q_adapt=unlabeled_soft_q_adapt,
            max_dim=32,
            unl_conf_thresh=0.85,
            unlabeled_mean_weight=0.10,
        )

        self._fit_radial_support_model(
            train_feats_adapt=train_feats_adapt,
            train_labels=train_labels_tensor,
        )

    @torch.no_grad()
    def _knn_posterior_logits(self, query_features):
        query_features = safe_normalize(sanitize_tensor(query_features), dim=1)
        if (
            self.memory_features is None
            or self.memory_posteriors is None
            or self.memory_features.shape[0] == 0
        ):
            return torch.zeros(
                (query_features.shape[0], self.num_classes),
                device=query_features.device,
                dtype=query_features.dtype,
            )

        memory = safe_normalize(sanitize_tensor(self.memory_features), dim=1)
        memory_post = sanitize_tensor(self.memory_posteriors)
        memory_rel = self.memory_reliability
        if memory_rel is None:
            memory_rel = torch.ones(
                memory.shape[0], device=memory.device, dtype=memory.dtype
            )

        k_eff = min(self.knn_k, memory.shape[0])
        sims = sanitize_tensor(query_features @ memory.T)
        vals, idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=False)
        weights = torch.softmax(vals / self.knn_temperature, dim=1)

        rel = memory_rel[idx]
        weights = weights * rel
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        neigh_post = memory_post[idx]
        class_post = (weights.unsqueeze(-1) * neigh_post).sum(dim=1)
        class_post = class_post / (class_post.sum(dim=1, keepdim=True) + 1e-12)
        return sanitize_tensor(torch.log(class_post + 1e-12))

    @torch.no_grad()
    def _retrieval_rerank_logits(self, query_features):
        query_features = safe_normalize(sanitize_tensor(query_features), dim=1)
        if (
            self.memory_features is None
            or self.memory_posteriors is None
            or self.rerank_support_strength is None
            or self.memory_features.shape[0] == 0
        ):
            return torch.zeros(
                (query_features.shape[0], self.num_classes),
                device=query_features.device,
                dtype=query_features.dtype,
            )

        memory = safe_normalize(sanitize_tensor(self.memory_features), dim=1)
        memory_post = sanitize_tensor(self.memory_posteriors)
        support = sanitize_tensor(self.rerank_support_strength)
        memory_rel = self.memory_reliability
        if memory_rel is None:
            memory_rel = torch.ones(
                memory.shape[0], device=memory.device, dtype=memory.dtype
            )

        k_eff = min(self.rerank_query_k, memory.shape[0])
        sims = sanitize_tensor(query_features @ memory.T)
        vals, idx = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=False)

        base_w = torch.softmax(vals / self.rerank_temperature, dim=1)
        trust = support[idx]
        rel = memory_rel[idx]
        weights = base_w * trust * rel
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

        neigh_post = memory_post[idx]
        class_post = (weights.unsqueeze(-1) * neigh_post).sum(dim=1)
        class_post = class_post / (class_post.sum(dim=1, keepdim=True) + 1e-12)
        return sanitize_tensor(torch.log(class_post + 1e-12))

    @torch.no_grad()
    def _top2_margin(self, logits):
        logits = sanitize_tensor(logits)
        if logits.shape[1] < 2:
            return torch.zeros(
                logits.shape[0], device=logits.device, dtype=logits.dtype
            )
        vals = torch.topk(logits, k=2, dim=1).values
        return sanitize_tensor(vals[:, 0] - vals[:, 1])

    @torch.no_grad()
    def _adaptive_temperature_scale(self, proto_logits, knn_logits, rerank_logits):
        proto_logits = sanitize_tensor(proto_logits)
        knn_logits = sanitize_tensor(knn_logits)
        rerank_logits = sanitize_tensor(rerank_logits)

        pm = self._top2_margin(proto_logits)
        km = self._top2_margin(knn_logits)
        rm = self._top2_margin(rerank_logits)

        p_pred = proto_logits.argmax(dim=1)
        k_pred = knn_logits.argmax(dim=1)
        r_pred = rerank_logits.argmax(dim=1)

        agree_pk = (p_pred == k_pred).float()
        agree_pr = (p_pred == r_pred).float()
        agree_kr = (k_pred == r_pred).float()

        conf = (
            0.45 * torch.tanh(2.5 * pm)
            + 0.30 * torch.tanh(2.5 * km)
            + 0.15 * torch.tanh(2.0 * rm)
            + 0.06 * agree_pk
            + 0.02 * agree_pr
            + 0.02 * agree_kr
        )

        conf = conf.clamp(0.0, 1.0)
        scale = (
            self.adaptive_temp_min
            + (self.adaptive_temp_max - self.adaptive_temp_min) * conf
        )
        return sanitize_tensor(scale.unsqueeze(1))

    def _make_tta_views(self, pil_img: Image.Image):
        size = self.input_size
        resize_size = max(size + 32, int(round(size * 1.12)))

        img = pil_img.convert("RGB")
        img_r = TF.resize(img, resize_size, interpolation=TF.InterpolationMode.BILINEAR)

        w, h = img_r.size
        if w < size or h < size:
            img_r = TF.resize(
                img_r, [size, size], interpolation=TF.InterpolationMode.BILINEAR
            )
            w, h = img_r.size

        top = 0
        left = 0
        bottom = max(h - size, 0)
        right = max(w - size, 0)
        center_top = max((h - size) // 2, 0)
        center_left = max((w - size) // 2, 0)

        crops = [
            TF.crop(img_r, center_top, center_left, size, size),
            TF.crop(img_r, top, left, size, size),
            TF.crop(img_r, bottom, right, size, size),
            TF.hflip(TF.crop(img_r, center_top, center_left, size, size)),
        ]

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        out = [normalize(TF.to_tensor(c)) for c in crops]
        return out

    @torch.no_grad()
    def _extract_adapted_features_from_tensor_batch(self, image_batch):
        image_batch = image_batch.to(self.device, non_blocking=True)
        base_features = self.model(image_batch)
        base_features = safe_normalize(sanitize_tensor(base_features), dim=1)

        base_features = self._query_smooth_against_memory(
            base_features,
            memory_features=self.base_memory_features,
            k=10,
            self_weight=0.7,
            temperature=0.07,
        )

        if self.metric_model is not None:
            features, _ = self.metric_model(base_features)
        else:
            features = base_features
        features = safe_normalize(sanitize_tensor(features), dim=1)

        features = self._query_smooth_against_memory(
            features,
            memory_features=self.memory_features,
            k=self.query_adapt_smooth_k,
            self_weight=self.query_adapt_self_weight,
            temperature=self.query_adapt_smooth_temperature,
        )
        return sanitize_tensor(features)

    @torch.no_grad()
    def _fuse_logits_from_features(self, features):
        features = safe_normalize(sanitize_tensor(features), dim=1)

        proto_logits = self._multi_proto_logits(features)
        knn_logits = self._knn_posterior_logits(features)
        rerank_logits = self._retrieval_rerank_logits(features)
        lda_logits = self._lda_logits(features)
        radial_logits = self._radial_support_logits(features)

        base_logits = (
            self.proto_blend_weight * proto_logits
            + (1.0 - self.proto_blend_weight) * knn_logits
        )
        logits = (
            base_logits
            + self.rerank_weight * rerank_logits
            + self.lda_weight * lda_logits
            + self.radial_weight * radial_logits
        )

        temp_scale = self._adaptive_temperature_scale(
            proto_logits=proto_logits,
            knn_logits=knn_logits,
            rerank_logits=rerank_logits,
        )
        logits = sanitize_tensor(logits * temp_scale)
        return logits

    @torch.no_grad()
    def _infer_from_tensor_batch(self, image_batch):
        features = self._extract_adapted_features_from_tensor_batch(image_batch)
        logits = self._fuse_logits_from_features(features)
        return logits, features

    @torch.no_grad()
    def _tta_view_consistency_correct(self, view_features, view_logits):
        if (not self.view_consistency_enabled) or len(view_features) <= 1:
            return [sanitize_tensor(v) for v in view_features]

        feats = torch.stack(
            [safe_normalize(sanitize_tensor(f), dim=1) for f in view_features], dim=0
        )
        logits = torch.stack([sanitize_tensor(l) for l in view_logits], dim=0)

        probs = torch.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values
        w = 0.25 + conf
        w = w / (w.sum(dim=0, keepdim=True) + 1e-12)

        consensus = (w.unsqueeze(-1) * feats).sum(dim=0)
        consensus = safe_normalize(consensus, dim=1)

        corrected = []
        for v in range(feats.shape[0]):
            fv = feats[v]
            agree = (fv * consensus).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            alpha = self.view_consensus_strength + self.view_low_agree_boost * (
                1.0 - agree
            )
            alpha = alpha.clamp(0.0, 0.45)

            f_corr = safe_normalize((1.0 - alpha) * fv + alpha * consensus, dim=1)

            if (
                self.memory_features is not None
                and self.view_memory_post_smooth_strength > 0
            ):
                mem_corr = self._query_smooth_against_memory(
                    f_corr,
                    memory_features=self.memory_features,
                    k=min(6, self.query_adapt_smooth_k),
                    self_weight=1.0 - self.view_memory_post_smooth_strength,
                    temperature=self.query_adapt_smooth_temperature,
                )
                f_corr = safe_normalize(mem_corr, dim=1)

            corrected.append(sanitize_tensor(f_corr))
        return corrected

    @torch.no_grad()
    def predict(self, image_batch):
        self.model.eval()

        if isinstance(image_batch, list):
            if len(image_batch) == 0:
                logits = torch.empty((0, self.num_classes), device=self.device)
                feats = torch.empty(
                    (0, self.metric_out_dim or self.feature_dim), device=self.device
                )
                return logits, feats, []

            per_view_tensors = []
            for img in image_batch:
                if torch.is_tensor(img):
                    if img.dim() == 3:
                        per_view_tensors.append([img])
                    else:
                        raise TypeError(
                            "Tensor items inside image list must be CHW tensors."
                        )
                else:
                    per_view_tensors.append(self._make_tta_views(img))

            n_views = len(per_view_tensors[0])
            if not all(len(vs) == n_views for vs in per_view_tensors):
                raise RuntimeError("Inconsistent number of TTA views.")

            view_logits = []
            view_features = []
            for v in range(n_views):
                batch_v = torch.stack(
                    [per_view_tensors[i][v] for i in range(len(per_view_tensors))],
                    dim=0,
                )
                feats_v = self._extract_adapted_features_from_tensor_batch(batch_v)
                logits_v = self._fuse_logits_from_features(feats_v)
                view_logits.append(sanitize_tensor(logits_v))
                view_features.append(sanitize_tensor(feats_v))

            corrected_features = self._tta_view_consistency_correct(
                view_features, view_logits
            )
            corrected_logits = [
                self._fuse_logits_from_features(fv) for fv in corrected_features
            ]

            logits_stack = sanitize_tensor(torch.stack(corrected_logits, dim=0))
            probs_stack = torch.softmax(logits_stack, dim=-1)
            conf = probs_stack.max(dim=-1).values

            weights = 0.25 + conf
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-12)
            logits = sanitize_tensor((weights.unsqueeze(-1) * logits_stack).sum(dim=0))
            features = sanitize_tensor(corrected_features[0])

        elif torch.is_tensor(image_batch):
            if image_batch.dim() == 3:
                image_batch = image_batch.unsqueeze(0)
            logits, features = self._infer_from_tensor_batch(image_batch)
        else:
            raise TypeError("image_batch must be a torch.Tensor or list of PIL images")

        logits = sanitize_tensor(logits)
        features = sanitize_tensor(features)
        pred_idx = torch.argmax(logits, dim=1).tolist()
        predictions = [{"category_id": self.idx_to_cat_id[p]} for p in pred_idx]
        return logits, features, predictions
