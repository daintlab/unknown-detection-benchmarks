import os
import matplotlib.pyplot as plt

def training_plot(trn_log, tst_log, save_path):

    # read log
    train_log = trn_log.read()
    test_log = tst_log.read()

    # zip log data
    epoch, train_loss, train_acc = zip(*train_log)
    epoch, test_loss, test_acc = zip(*test_log)

    # train & valid cross entropy loss
    plt.plot(epoch, train_loss, '-b', label='train')
    plt.plot(epoch, test_loss, '-r', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(loc='upper right')
    plt.title(f'CIFAR40 Loss: {test_loss[-1]}')
    plt.savefig(os.path.join(save_path, 'cifar40-loss.png'))
    plt.close()

    # train & valid acc
    plt.plot(epoch, train_acc, '-b', label='train')
    plt.plot(epoch, test_acc, '-r', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.title(f'CIFAR40 Test Accuracy: {test_acc[-1]}')
    plt.savefig(os.path.join(save_path, 'cifar40-accuracy.png'))
    plt.close()
