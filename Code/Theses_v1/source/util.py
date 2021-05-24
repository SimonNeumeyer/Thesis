from enum import Enum
from pathlib import Path
import datetime

class Constants():
    ALPHA = "ALPHA"
    REDUCE_FUNC_SUM = "REDUCE_FUNC_SUM"
    PATH_DATA = Path("../data")
    
    @staticmethod
    def path_tensorboard():
        return Path(f"../runs/{datetime.datetime.now().strftime('%b%d%H%M%S')}")

class OptimizationSettings():
    
    def __init__(self, **kwargs):
        self._init_default()
        self.update(**kwargs)
        
    def _init_default(self):
        self.alpha_update = True
        self.alpha_sampling = False
        self.diffNN_reduce = Constants.REDUCE_FUNC_SUM
        self.graphNN_reduce = Constants.REDUCE_FUNC_SUM
        self.graphNN_normalize = True
        self.diffNN_normalize = False
        
    def update(self, **kwargs):
        for key in kwargs:
            assert key in self.__dict__, f"Setting '{key}' not available"
            self.__dict__[key] = kwargs[key]