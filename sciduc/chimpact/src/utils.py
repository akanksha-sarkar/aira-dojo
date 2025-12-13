"""
Standalone utilities for pose dataset without mmengine dependencies.
"""
import os.path as osp
import warnings
import json
import numpy as np


class Config:
    """Simple config class to replace mmengine.Config."""
    
    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = cfg_dict
    
    @classmethod
    def fromfile(cls, filename):
        """Load config from file."""
        if not osp.isfile(filename):
            raise FileNotFoundError(f'Config file {filename} does not exist')
        
        with open(filename, 'r') as f:
            if filename.endswith('.py'):
                # For Python config files, we'll need to execute them
                # This is a simplified version - in practice you might want more robust parsing
                exec_globals = {}
                exec(f.read(), exec_globals)
                cfg_dict = exec_globals.get('dataset_info', {})
                if not cfg_dict:
                    raise ValueError(
                        f'Config file {filename} does not contain "dataset_info" '
                        f'or it is empty')
            elif filename.endswith('.json'):
                cfg_dict = json.load(f)
            else:
                raise ValueError(f'Unsupported config file format: {filename}')
        
        if not cfg_dict:
            raise ValueError(f'Config file {filename} contains no data')
        
        return cls(cfg_dict)
    
    def __getattr__(self, name):
        if name in self._cfg_dict:
            return self._cfg_dict[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key):
        return self._cfg_dict[key]
    
    def __setitem__(self, key, value):
        self._cfg_dict[key] = value
    
    def __contains__(self, key):
        """Check if key is in config."""
        return key in self._cfg_dict
    
    def keys(self):
        """Return keys of config dict."""
        return self._cfg_dict.keys()
    
    def get(self, key, default=None):
        """Get value from config with default."""
        return self._cfg_dict.get(key, default)
    
    def to_dict(self):
        """Convert Config to dict."""
        return self._cfg_dict.copy()


def parse_pose_metainfo(metainfo: dict):
    """Load meta information of pose dataset and check its integrity.

    Args:
        metainfo (dict): Raw data of pose meta information, which should
            contain following contents:

            - "dataset_name" (str): The name of the dataset
            - "keypoint_info" (dict): The keypoint-related meta information,
                e.g., name, upper/lower body, and symmetry
            - "skeleton_info" (dict): The skeleton-related meta information,
                e.g., start/end keypoint of limbs
            - "joint_weights" (list[float]): The loss weights of keypoints
            - "sigmas" (list[float]): The keypoint distribution parameters
                to calculate OKS score. See `COCO keypoint evaluation
                <https://cocodataset.org/#keypoints-eval>`__.

            An example of metainfo is shown as follows.

            .. code-block:: none
                {
                    "dataset_name": "coco",
                    "keypoint_info":
                    {
                        0:
                        {
                            "name": "nose",
                            "type": "upper",
                            "swap": "",
                            "color": [51, 153, 255],
                        },
                        1:
                        {
                            "name": "right_eye",
                            "type": "upper",
                            "swap": "left_eye",
                            "color": [51, 153, 255],
                        },
                        ...
                    },
                    "skeleton_info":
                    {
                        0:
                        {
                            "link": ("left_ankle", "left_knee"),
                            "color": [0, 255, 0],
                        },
                        ...
                    },
                    "joint_weights": [1., 1., ...],
                    "sigmas": [0.026, 0.025, ...],
                }


            A special case is that `metainfo` can have the key "from_file",
            which should be the path of a config file. In this case, the
            actual metainfo will be loaded by:

            .. code-block:: python
                metainfo = Config.fromfile(metainfo['from_file'])

    Returns:
        Dict: pose meta information that contains following contents:

        - "dataset_name" (str): Same as ``"dataset_name"`` in the input
        - "num_keypoints" (int): Number of keypoints
        - "keypoint_id2name" (dict): Mapping from keypoint id to name
        - "keypoint_name2id" (dict): Mapping from keypoint name to id
        - "upper_body_ids" (list): Ids of upper-body keypoint
        - "lower_body_ids" (list): Ids of lower-body keypoint
        - "flip_indices" (list): The Id of each keypoint's symmetric keypoint
        - "flip_pairs" (list): The Ids of symmetric keypoint pairs
        - "keypoint_colors" (numpy.ndarray): The keypoint color matrix of
            shape [K, 3], where each row is the color of one keypint in bgr
        - "num_skeleton_links" (int): The number of links
        - "skeleton_links" (list): The links represented by Id pairs of start
             and end points
        - "skeleton_link_colors" (numpy.ndarray): The link color matrix
        - "dataset_keypoint_weights" (numpy.ndarray): Same as the
            ``"joint_weights"`` in the input
        - "sigmas" (numpy.ndarray): Same as the ``"sigmas"`` in the input
    """

    if 'from_file' in metainfo:
        cfg_file = metainfo['from_file']
        
        # Try to resolve relative paths
        if not osp.isabs(cfg_file) and not osp.isfile(cfg_file):
            # Try relative to current directory
            possible_paths = [
                cfg_file,
                osp.join(osp.dirname(__file__), cfg_file),
            ]
            for path in possible_paths:
                if osp.isfile(path):
                    cfg_file = path
                    break
        
        if not osp.isfile(cfg_file):
            # For standalone version, we'll create a default config if file doesn't exist
            warnings.warn(
                f'The metainfo config file "{cfg_file}" does not exist. '
                f'Using default LeipzigChimp configuration.')
            # Create default LeipzigChimp metainfo
            metainfo = create_default_leipzigchimp_metainfo()
        else:
            try:
                config = Config.fromfile(cfg_file)
                # Convert Config object to dict for compatibility
                if isinstance(config, Config):
                    metainfo = config.to_dict()
                else:
                    metainfo = config
            except Exception as e:
                warnings.warn(
                    f'Failed to load config file "{cfg_file}": {e}. '
                    f'Using default LeipzigChimp configuration.')
                metainfo = create_default_leipzigchimp_metainfo()

    # check data integrity
    assert 'dataset_name' in metainfo
    assert 'keypoint_info' in metainfo
    assert 'skeleton_info' in metainfo
    assert 'joint_weights' in metainfo
    assert 'sigmas' in metainfo

    # parse metainfo
    parsed = dict(
        dataset_name=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed['dataset_name'] = metainfo['dataset_name']

    # parse keypoint information
    parsed['num_keypoints'] = len(metainfo['keypoint_info'])

    for kpt_id, kpt in metainfo['keypoint_info'].items():
        kpt_name = kpt['name']
        parsed['keypoint_id2name'][kpt_id] = kpt_name
        parsed['keypoint_name2id'][kpt_name] = kpt_id
        parsed['keypoint_colors'].append(kpt.get('color', [255, 128, 0]))

        kpt_type = kpt.get('type', '')
        if kpt_type == 'upper':
            parsed['upper_body_ids'].append(kpt_id)
        elif kpt_type == 'lower':
            parsed['lower_body_ids'].append(kpt_id)

        swap_kpt = kpt.get('swap', '')
        if swap_kpt == kpt_name or swap_kpt == '':
            parsed['flip_indices'].append(kpt_name)
        else:
            parsed['flip_indices'].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed['flip_pairs']:
                parsed['flip_pairs'].append(pair)

    # parse skeleton information
    parsed['num_skeleton_links'] = len(metainfo['skeleton_info'])
    for _, sk in metainfo['skeleton_info'].items():
        parsed['skeleton_links'].append(sk['link'])
        parsed['skeleton_link_colors'].append(sk.get('color', [96, 96, 255]))

    # parse extra information
    parsed['dataset_keypoint_weights'] = np.array(
        metainfo['joint_weights'], dtype=np.float32)
    parsed['sigmas'] = np.array(metainfo['sigmas'], dtype=np.float32)

    # formatting
    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]

    parsed['flip_pairs'] = _map(
        parsed['flip_pairs'], mapping=parsed['keypoint_name2id'])
    parsed['flip_indices'] = _map(
        parsed['flip_indices'], mapping=parsed['keypoint_name2id'])
    parsed['skeleton_links'] = _map(
        parsed['skeleton_links'], mapping=parsed['keypoint_name2id'])

    parsed['keypoint_colors'] = np.array(
        parsed['keypoint_colors'], dtype=np.uint8)
    parsed['skeleton_link_colors'] = np.array(
        parsed['skeleton_link_colors'], dtype=np.uint8)

    return parsed


def create_default_leipzigchimp_metainfo():
    """Create default LeipzigChimp metainfo."""
    return {
        'dataset_name': 'leipzigchimp',
        'keypoint_info': {
            0: {'name': 'pelvis', 'type': 'lower', 'swap': '', 'color': [255, 128, 0]},
            1: {'name': 'right_knee', 'type': 'lower', 'swap': 'left_knee', 'color': [255, 128, 0]},
            2: {'name': 'right_ankle', 'type': 'lower', 'swap': 'left_ankle', 'color': [255, 128, 0]},
            3: {'name': 'left_knee', 'type': 'lower', 'swap': 'right_knee', 'color': [255, 128, 0]},
            4: {'name': 'left_ankle', 'type': 'lower', 'swap': 'right_ankle', 'color': [255, 128, 0]},
            5: {'name': 'neck', 'type': 'upper', 'swap': '', 'color': [255, 128, 0]},
            6: {'name': 'upper_lip', 'type': 'upper', 'swap': '', 'color': [255, 128, 0]},
            7: {'name': 'lower_lip', 'type': 'upper', 'swap': '', 'color': [255, 128, 0]},
            8: {'name': 'right_eye', 'type': 'upper', 'swap': 'left_eye', 'color': [255, 128, 0]},
            9: {'name': 'left_eye', 'type': 'upper', 'swap': 'right_eye', 'color': [255, 128, 0]},
            10: {'name': 'right_shoulder', 'type': 'upper', 'swap': 'left_shoulder', 'color': [255, 128, 0]},
            11: {'name': 'right_elbow', 'type': 'upper', 'swap': 'left_elbow', 'color': [255, 128, 0]},
            12: {'name': 'right_wrist', 'type': 'upper', 'swap': 'left_wrist', 'color': [255, 128, 0]},
            13: {'name': 'left_shoulder', 'type': 'upper', 'swap': 'right_shoulder', 'color': [255, 128, 0]},
            14: {'name': 'left_elbow', 'type': 'upper', 'swap': 'right_elbow', 'color': [255, 128, 0]},
            15: {'name': 'left_wrist', 'type': 'upper', 'swap': 'right_wrist', 'color': [255, 128, 0]},
        },
        'skeleton_info': {
            0: {'link': ('pelvis', 'right_knee'), 'color': [0, 255, 0]},
            1: {'link': ('right_knee', 'right_ankle'), 'color': [0, 255, 0]},
            2: {'link': ('pelvis', 'left_knee'), 'color': [0, 255, 0]},
            3: {'link': ('left_knee', 'left_ankle'), 'color': [0, 255, 0]},
            4: {'link': ('pelvis', 'neck'), 'color': [0, 255, 0]},
            5: {'link': ('neck', 'upper_lip'), 'color': [0, 255, 0]},
            6: {'link': ('upper_lip', 'lower_lip'), 'color': [0, 255, 0]},
            7: {'link': ('neck', 'right_eye'), 'color': [0, 255, 0]},
            8: {'link': ('neck', 'left_eye'), 'color': [0, 255, 0]},
            9: {'link': ('neck', 'right_shoulder'), 'color': [0, 255, 0]},
            10: {'link': ('right_shoulder', 'right_elbow'), 'color': [0, 255, 0]},
            11: {'link': ('right_elbow', 'right_wrist'), 'color': [0, 255, 0]},
            12: {'link': ('neck', 'left_shoulder'), 'color': [0, 255, 0]},
            13: {'link': ('left_shoulder', 'left_elbow'), 'color': [0, 255, 0]},
            14: {'link': ('left_elbow', 'left_wrist'), 'color': [0, 255, 0]},
        },
        'joint_weights': [1.0] * 16,
        'sigmas': [0.026] * 16,
    }



