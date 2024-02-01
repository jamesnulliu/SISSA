import numpy as np
import pandas as pd
from .attacks import *
from .hardware_failures import *
from .. import utils

def normal(packets:pd.DataFrame, **kwargs) -> pd.DataFrame:
    return packets

class Manipulator:
    def __init__(self):
        self.idx2method = {
            0: normal,
            1: DDos,
            2: FakeInterface,
            3: FakeSource,
            4: RequestWithoutResponse,
            5: ResponseWithoutRequest,
            6: WeibullFailure
        }
    
    def manipulate(self, packets:pd.DataFrame, idx:int, **kwargs) -> pd.DataFrame:
        packets = packets.reset_index(drop=True)
        packets = self.idx2method[idx](packets, **kwargs)
        return packets