import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback

#this class visualase training process in real-time
class RealTimeMonitoring(Callback):
    def __init__(self):
        super(RealTimeMonitoring, self).__init__()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
    # this fucntion is called on the end of each epoch to update the graphs
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        # plot the results
        plt.figure(figsize=(12,6))
        #Loss plot
        plt.subplot(1,2,1)
        plt.plot(self.train_loss, label='Training Loss', color='blue')
        plt.plot(self.val_loss, label='Validation Loss', color='red')
        plt.title('Real-Time Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        #Accuracy plot
        plt.subplot(1,2,2)
        plt.plot(self.train_acc, label='Training Accuracy', color='blue')
        plt.plot(self.val_acc, label='Validation Accuracy', color='red')
        plt.title('Real-Time Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.pause(0.01)
        plt.show(block=False)