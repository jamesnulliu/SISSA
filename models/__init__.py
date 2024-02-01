from .sissa_cnn import SISSA_CNN
from .sissa_rnn import SISSA_RNN
from .sissa_lstm import SISRA_LSTM

SISSA_MODELS = {
    "SISSA_CNN": SISSA_CNN,
    "SISSA_RNN": SISSA_RNN,
    "SISSA_LSTM": SISRA_LSTM
}

__all__ = ["SISSA_CNN", "SISSA_RNN", "SISRA_LSTM", "SISSA_MODELS"]
