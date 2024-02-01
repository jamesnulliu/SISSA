import initialize
from preprocess import Preprocessor

if __name__ == "__main__":
    initialize.init()
    preprocessor = Preprocessor()
    # preprocessor.preprocess(rd=5)
    preprocessor.train_val_split()