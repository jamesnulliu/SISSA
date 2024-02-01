import numpy as np
import matplotlib.pyplot as plt
import os


def plot_loss_acc():
    paths = [
        r"results/sisra_cnn/train_losses.npy",
        r"results/sisra_cnn/val_accs.npy",
        r"results/sisra_cnn_a/train_losses.npy",
        r"results/sisra_cnn_a/val_accs.npy",
    ]
    names = [
        "SISSA-C Train Loss",
        "SISSA-C Val Acc",
        "SISSA-C-A Train Loss",
        "SISSA-C-A Val Acc",
    ]
    # paths = [
    #     r"results/sisra_rnn/train_losses.npy",
    #     r"results/sisra_rnn/val_accs.npy",
    #     r"results/sisra_rnn_a/train_losses.npy",
    #     r"results/sisra_rnn_a/val_accs.npy",
    # ]
    # names = [
    #     "SISSA-R Train Loss",
    #     "SISSA-R Val Acc",
    #     "SISSA-R-A Train Loss",
    #     "SISSA-R-A Val Acc",
    # ]
    # paths = [
    #     r"results/sisra_lstm/train_losses.npy",
    #     r"results/sisra_lstm/val_accs.npy",
    #     r"results/sisra_lstm_a/train_losses.npy",
    #     r"results/sisra_lstm_a/val_accs.npy",
    # ]
    # names = [
    #     "SISSA-LTrain Loss",
    #     "SISSA-L Val Acc",
    #     "SISSA-L-A Train Loss",
    #     "SISSA-L-A Val Acc",
    # ]

    loss_1 = np.load(file=paths[0])
    acc_1 = np.load(file=paths[1])
    loss_2 = np.load(file=paths[2])
    acc_2 = np.load(file=paths[3])

    acc_range = [0, 1]
    loss_range = [0, 10]

    # Create a figure with a specific size
    fig = plt.figure(figsize=(10, 10))

    # Create an axis instance
    ax1 = fig.add_subplot(111)

    # Data for the two lines
    x = range(0, int(len(loss_1)))
    # Plot accuracy
    line1, = ax1.plot(x, acc_1, label=names[1]) 
    line2, = ax1.plot(x, acc_2, label=names[3])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(acc_range)
    # Plot loss
    ax2 = ax1.twinx()
    line3, = ax2.plot(x, loss_1, label=names[0])
    line4, = ax2.plot(x, loss_2, label=names[2])
    ax2.set_ylabel("Loss")
    ax2.set_ylim(loss_range)  # Set the y-range for the second line

    plt.legend(handles=[line1, line3, line2, line4], loc="upper left")
    plt.savefig("results/sisra_cnn/acc-loss.png")


def count_for_each_class(data_path: str, label_path: str):

    labels = np.load(file=label_path)
    # Count occurences of each class in the dataset
    unique, counts = np.unique(labels, return_counts=True)
    # Print the counts
    print(dict(zip(unique, counts)))
    print("Total number of samples: ", len(labels))




def main():
    plot_loss_acc()


if __name__ == "__main__":
    main()
