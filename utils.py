
import json
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback

class LinearDecayLR(Callback):
    '''
    A simple callback to reduce learning rate linearly after each batch update. 
    
    # Arguments
        min_lr: The lower bound (final value) of the learning rate.
        max_lr: The upper bound (initial value) of the learning rate.
        steps_per_epoch: Number of mini-batches for an epoch. 
        epochs: Number of epochs to run training. 
        
    # Usage
        lr_decay = LinearDecayLR(min_lr=1e-5, max_lr=0.01, 
                                 steps_per_epoch=step_size_train, 
                                 epochs=20, verbose=1)
        model.fit(X_train, Y_train, callbacks=[lr_decay])
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None, verbose=0):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.verbose = verbose
        
    def linear_decay(self):
        '''Calculate the learning rate.'''
        r = self.iteration / self.total_iterations 
        return self.max_lr - (self.max_lr-self.min_lr) * r
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the initial value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Update the learning rate after each batch update'''
        logs = logs or {}
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, self.linear_decay())
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, K.get_value(self.model.optimizer.lr)))    
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

def save_model_parms(data, fname='./model_parms.json'):
    '''
    Save model parameters stored in a dictionary as json file. 
    '''
    with open(fname, 'w') as fp:
        json.dump(data, fp)

def load_model_parms(fname='./model_parms.json'):
    '''
    Load model parameters from json file to be user for prediction. 
    '''
    with open(fname) as data_file:
        data = json.load(data_file)
    return data

def save_training_history(history_dict, fname='./training_history.npy'):
    '''
    Save training history returned from model.fit_generator(). 
    '''
    np.save(fname, history_dict)
    return

def load_training_history(fname='./training_history.npy'):
    '''
    Load training history returned from model.fit_generator(). 
    '''
    history_obj = np.load(fname)
    history_dict = history_obj.item()
    return history_dict

def plot_training_history(history_dict):
    '''
    Plot training history of accuracy, loss and learning rate. 
    '''
    epochs = np.arange(1, len(history_dict['acc']) + 1)
    fig = plt.figure()
    # Plot train/val accuracy
    plt.subplot(3, 1, 1)
    plt.plot(epochs, history_dict['acc'], 'bo-', label='Train Acc')
    plt.plot(epochs, history_dict['val_acc'], 'ro-', label='Val Acc')
    plt.legend()
    plt.title('Training History')
    plt.ylabel('Accuracy')
    plt.xlim(0,len(history_dict['acc']) + 1)
    # Plot train/val loss
    plt.subplot(3, 1, 2)
    plt.plot(epochs, history_dict['loss'], 'bo-', label='Train Loss')
    plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Val Loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlim(0,len(history_dict['acc']) + 1)
    # Plot learning rate
    plt.subplot(3, 1, 3)
    plt.plot(epochs, history_dict['lr'], 'ko-')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.xlim(0,len(history_dict['acc']) + 1)
    plt.show()
    return
    