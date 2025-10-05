# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dojo.tasks.mlebench.task import MLEBenchTask
from dojo.tasks.detection.task import DetectionTask

TASK_MAP = {
  "MLEBenchTaskConfig": MLEBenchTask,
  "DetectionConfig": DetectionTask
  }
