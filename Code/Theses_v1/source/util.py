from pathlib import Path
from uuid import uuid4
import datetime

class Constants():
    ALPHA = "ALPHA"
    REDUCE_FUNC_SUM = "REDUCE_FUNC_SUM"
    OPTIMIZER_ADAM = "ADAM"
    PATH_DATA = Path("../data")
    ACTIVATION_RELU = "ACTIVATION_RELU"
    OPERATION_LINEAR = "OPERATION_LINEAR"
    OPERATION_CONV = "OPERATION_CONV"
    POOLING_MAX = "POOLING_MAX"
    POOLING_AVG = "POOLING_AVG"

def uuid(time=True):
    if time:
        return datetime.datetime.now().strftime('%b%d%H%M') + str(uuid4()).translate({ord(c): "" for c in "-"})
    else:
        return str(uuid4()).translate({ord(c): "" for c in "-"})
    
def path_tensorboard(args=""):
    return Path(f"../runs")

def copy_tensor(tensor):
    return tensor.data.clone()