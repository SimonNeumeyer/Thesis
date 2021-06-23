from pathlib import Path
import datetime

class Constants():
    ALPHA = "ALPHA"
    REDUCE_FUNC_SUM = "REDUCE_FUNC_SUM"
    PATH_DATA = Path("../data")
    
def path_tensorboard(args):
    return Path(f"../runs/{datetime.datetime.now().strftime('%b%d%H%M') + args}")

def copy_tensor(tensor):
    return tensor.data.clone()