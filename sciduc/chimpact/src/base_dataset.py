"""
Standalone Base Dataset implementation without mmengine dependencies.
"""
import copy
import os.path as osp
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import json
import numpy as np


class BaseDataset:
    """Base class for datasets without mmengine dependencies.
    
    Args:
        ann_file (str): Annotation file path. Default: ''.
        metainfo (dict, optional): Meta information for dataset. Default: None.
        data_root (str, optional): The root directory for data_prefix and ann_file. Default: None.
        data_prefix (dict, optional): Prefix for training data. Default: dict(img='').
        filter_cfg (dict, optional): Config for filter data. Default: None.
        indices (int or Sequence[int], optional): Support using first few data in annotation file. Default: None.
        serialize_data (bool, optional): Whether to hold memory using serialized objects. Default: True.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): test_mode=True means in test phase. Default: False.
        lazy_init (bool, optional): Whether to load annotation during instantiation. Default: False.
        max_refetch (int, optional): Maximum extra number of cycles to get a valid image. Default: 1000.
    """
    
    METAINFO: dict = dict()
    
    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        
        self.ann_file = ann_file
        self.data_root = data_root
        self.data_prefix = data_prefix
        self.filter_cfg = filter_cfg
        self.indices = indices
        self.serialize_data = serialize_data
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        
        # Load metainfo
        self._metainfo = self._load_metainfo(metainfo)
        
        # Load data list
        if not lazy_init:
            self._data_list = self.load_data_list()
        else:
            self._data_list = []
    
    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta."""
        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)
        
        if not isinstance(metainfo, dict):
            raise TypeError(f'metainfo should be a dict, but got {type(metainfo)}')
        
        return metainfo
    
    @property
    def metainfo(self) -> dict:
        """Get meta information."""
        return self._metainfo
    
    @property
    def data_list(self) -> List[dict]:
        """Get data list."""
        return self._data_list
    
    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self._data_list)
    
    def __getitem__(self, idx: int) -> Any:
        """Get data by index."""
        return self.get_data_info(idx)
    
    def get_data_info(self, idx: int) -> dict:
        """Get data info by index."""
        if idx >= len(self._data_list):
            raise IndexError(f'Index {idx} out of range for dataset of length {len(self._data_list)}')
        return self._data_list[idx]
    
    def load_data_list(self) -> List[dict]:
        """Load data list from annotation file."""
        raise NotImplementedError
    
    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg."""
        return self._data_list


def exists(file_path: str) -> bool:
    """Check if file exists."""
    return osp.exists(file_path)


def get_local_path(file_path: str):
    """Get local path for file."""
    class LocalPathContext:
        def __init__(self, path):
            self.path = path
        
        def __enter__(self):
            return self.path
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return LocalPathContext(file_path)


def load(file_path: str) -> Any:
    """Load data from file."""
    if not exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        else:
            return f.read()


def is_list_of(input_list: list, item_type: type) -> bool:
    """Check if input_list is a list of item_type."""
    if not isinstance(input_list, list):
        return False
    return all(isinstance(item, item_type) for item in input_list)


def force_full_init(func):
    """Decorator to ensure full initialization."""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_data_list') or not self._data_list:
            self._data_list = self.load_data_list()
        return func(self, *args, **kwargs)
    return wrapper
