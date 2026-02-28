# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Lazy imports to avoid circular import issues
def _get_task_map():
    # from dojo.tasks.mlebench.task import MLEBenchTask
    # from dojo.tasks.detection.task import DetectionTask
    from dojo.tasks.sciduc.task import SciDucTask
    from dojo.tasks.agentSSL.task import AgentSSLTask
    return {
        # "MLEBenchTaskConfig": MLEBenchTask,
        "SciDucConfig": SciDucTask,
        "AgentSSLConfig": AgentSSLTask
    }

# Use a property to make TASK_MAP lazy
class _TaskMap:
    def __getitem__(self, key):
        return _get_task_map()[key]
    
    def __contains__(self, key):
        return key in _get_task_map()
    
    def keys(self):
        return _get_task_map().keys()
    
    def values(self):
        return _get_task_map().values()
    
    def items(self):
        return _get_task_map().items()

TASK_MAP = _TaskMap()
