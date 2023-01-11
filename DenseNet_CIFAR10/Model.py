### YOUR CODE HEREtorch
# import tensorflow as tf
# import torch
import os, time
import numpy as np
from Network import DenseNet3
from ImageUtils import parse_record
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
from tqdm import tqdm
import sys
import pandas as pd

"""This script defines the training, validation and testing process.
"""


class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network =DenseNet3()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.network.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.configs['learning_rate'], weight_decay=self.configs['weight_decay'], momentum=0.91)
        self.max_epoch=self.configs['max_epoch']
        self.batch_size=self.configs['batch_size']
        self.modeldir= sys.path[0] + self.configs['model_dir']
        self.checkpt=self.configs['checkpts']


    def model_setup(self):
        pass

    def train(self, x_train, y_train, x_valid=None, y_valid=None):

        self.network.train()
        
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.batch_size

        
        training_loss=[]
        print('### Training... ###')
        for epoch in range(1, self.max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            print('epoch',epoch)
            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if epoch>0 and (epoch%70==0 or epoch%100==0 or epoch%170==0):
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']/10.
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay

                x_batch_train = curr_x_train[i*self.batch_size:min((i+1)*self.batch_size,curr_x_train.shape[0])]
                y_batch_train = curr_y_train[i*self.batch_size:min((i+1)*self.batch_size,curr_y_train.shape[0])]
                x_batch_train = np.array(list(map(lambda x: parse_record(x,True),x_batch_train))) # gotta check whether a lambda works as a map function over numpy array
                x_batch_train = torch.tensor(x_batch_train,device=self.device, dtype=torch.float)

                predictions = self.network.forward(x_batch_train)
                y_batch_train = torch.tensor(y_batch_train,device=self.device)
                y_batch_train=y_batch_train.to(torch.long)
                
                loss = self.loss_fn(predictions,y_batch_train)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time

            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
            training_loss.append([epoch,loss,duration])

            if epoch % self.configs['save_interval'] == 0:
                self.save(epoch)
        training_loss=pd.DataFrame(training_loss,columns=['epoch','loss','duration'])
        training_loss.to_csv(self.configs['iter']+'training_loss.csv')

    def evaluate(self, x, y, data_type):
        acc=[]
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in self.checkpt:
            checkpointfile = os.path.join(self.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            with torch.no_grad():
                for i in tqdm(range(x.shape[0])):
                    
                    ### YOUR CODE HERE
                    #processing the x before test pred
                    test_x_proc = np.array(list(map(lambda x: parse_record(x,False),x[i:i+1])))
                    test_x_proc = torch.tensor(test_x_proc,device=self.device,dtype=torch.float)
                    
                    preds.append(torch.argmax(self.network.forward(test_x_proc),axis=1))
                    ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
            acc.append(['{:.4f}'.format(torch.sum(preds==y)/y.shape[0]),checkpoint_num])
        acc=pd.DataFrame(acc, columns=['accuracy','epoch'])
        acc.to_csv(self.modeldir+self.configs['iter']+'_'+data_type+'_accuracy_1.csv')
        
        
        

    def save(self, epoch):
        checkpoint_path = os.path.join( self.modeldir, 'model-%d.ckpt'%(epoch))
        # os.makedirs(self.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def predict_prob(self, x,checkpoint_num):
        
        # self.network.eval()
        
        checkpointfile = os.path.join(self.modeldir, 'model-%d.ckpt'%(checkpoint_num))
        #checkpointfile = os.path.join(self.modeldir, 'model_final.ckpt')
        self.load(checkpointfile)
        self.network.eval()
        
        preds = []
        preds_check=[]
        with torch.no_grad():
            for i in tqdm(range(x.shape[0])):

                ### YOUR CODE HERE
                #processing the x before test pred
                test_x_proc = np.array(list(map(lambda x: parse_record(x,False),x[i:i+1])))
                test_x_proc = torch.tensor(test_x_proc,device=self.device,dtype=torch.float)

                # temp=torch.argmax(self.network.forward(test_x_proc),axis=1)
                
                
                pred = self.network.forward(test_x_proc)
                temp=torch.nn.functional.softmax(pred).cpu().detach().numpy().reshape((10))
                # temp=pred.cpu().detach().numpy().reshape((10))
                preds.append(temp)
                preds_check.append(np.argmax(temp))

        preds=pd.DataFrame(preds)
        preds.to_csv(self.modeldir+'_preds102.csv')
        preds_check=pd.DataFrame(preds_check)
        preds_check.to_csv(self.modeldir+'_check_preds102.csv')
        return preds


### END CODE HERE
