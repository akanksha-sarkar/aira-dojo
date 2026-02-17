# Adding pseudolabels to SciDUC task.

Agent now has write access to a pseudolabel directory /pseudolabel which will persist across runs.

TODO:

1. Modify evaluate to combine pseudolabel.json w/ annotation.json as the training dataset input.
2. Provide agents w/ unique unlabeled.json which has cocoformat image key. 
3. Remove images from unlabeled.json as they get added to pseudolabel.json


