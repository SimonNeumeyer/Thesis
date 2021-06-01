from pathlib import Path
import datetime

class Constants():
    ALPHA = "ALPHA"
    REDUCE_FUNC_SUM = "REDUCE_FUNC_SUM"
    PATH_DATA = Path("../data")
    
def path_tensorboard():
    return Path(f"../runs/{datetime.datetime.now().strftime('%b%d%H%M%S')}")

def copy_tensor(tensor):
    return tensor.data.clone()