#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:23:37 2018

@author: likun
"""

import pandas as pd
import numpy as np
import sklearn.neural_network as neural_network
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

class trainer:
  
  def __init__(self):
      self.train_x = np.zeros(1)
      self.train_y = np.zeros(1)
      self.test_x = np.zeros(1)
      self.test_y = np.zeros(1)
        
  #def load_data(train_x_file,train_y_file,test_x_file,test_y_file):
  def load_data(self,train_x_file):
      print("-"*30)
      print("starting to load the data")
      print("-"*30)
      self.train_x=pd.read_csv(train_x_file, sep=',',header=None)
      #self.train_y=pd.read_csv(train_y_file, sep=',',header=None)
      #self.test_x=pd.read_csv(test_x_file, sep=',',header=None)
      #self.test_y=pd.read_csv(test_y_file, sep=',',header=None)
      print("-"*30)
      print("reading finished")
      print("-"*30)
      #convert pandas data frame to numpy data frame
      #self.train_x = train_x.values
      #self.train_y = train_y.values
      
      #self.test_x = test_x.values
      #self.test_y = test_y.values
      print("-"*30)
      print("training set and test set are ready to analyze")
      print("-"*30)
      
      
  def accuracy(self,labels, predictions):
      return ((labels == predictions).astype(int)).mean()
    
  def preprocess(self,train_x,test_x):
      print("-"*30)
      print("preprocess the data")
      print("-"*30)
      self.train_x = preprocessing.normalize(train_x, norm='l2')     
      #self.test_x = preprocessing.normalize(test_x, norm='l2')
      print("-"*30)
      print("finished the data pre-processing")
      print("-"*30)
    
  def train_neural_net(self,train_x, train_y, test_x, test_y, hidden_layer):
      nn = neural_network.MLPClassifier(hidden_layer_sizes = hidden_layer)
      print("-"*30)
      print("fitting the neural network")
      print("-"*30)
      nn.fit(train_x,train_y)
      predictions = nn.predict(test_x)
      scores = self.accuracy(test_y,predictions)
      print("-"*30)
      print("the neural network achieves accuracy of :", scores)
      print("-"*30)
      
  def train_log_reg(self,train_x, train_y, test_x, test_y):
      lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
      print("-"*30)
      print("fitting the logistic regression")
      print("-"*30)
      train_y = np.greater(train_y,100*np.ones(train_y.shape)).astype(int)
      test_y = np.greater(test_y,100*np.ones(test_y.shape)).astype(int)
      lg.fit(train_x,train_y)
      predictions = lg.predict(test_x)
      scores = self.accuracy(test_y,predictions)
      print("-"*30)
      print("the logistic regression achieves accuracy of :", scores)
      print("-"*30)
      
if __name__ == '__main__':
   trainer = trainer()
   trainer.load_data("Test2_dataSet_Cpp.csv")
   train_x = pd.DataFrame(trainer.train_x)
   train_sp_y = train_x.iloc[1:8000,11:12].values
   train_sp_x=train_x.iloc[1:8000,1:11].values
   test_sp_y = train_x.iloc[8000:9000,11:12].values
   test_sp_x=train_x.iloc[8000:9000,1:11].values
   #train_x=dataset.iloc[:, 0:4].values
   trainer.preprocess(train_sp_x,test_sp_x)
   #train_sp_y = np.array(train_sp_y,float)
   #train_sp_y = np.array(train_sp_y,float)
# =============================================================================
#    for i in train_sp_y:
#        for j in i:
#           #print(type(i))
#           j = float(j)
#    for i in train_sp_y:
#        for j in i:
#           #print(type(i))
#           print(type(j))
#    for i in test_sp_y:
#        for j in i:
#           #print(type(i))
#           j = float(j)
# =============================================================================
   train_fl_y = np.zeros(train_sp_y.shape[0])
   h = 0
   for i in train_sp_y:
        for j in i:
           #print(type(i))
           train_fl_y[h] = int(j)
        h += 1
        
   test_fl_y = np.zeros(test_sp_y.shape[0])
   h = 0
   for i in test_sp_y:
        for j in i:
           #print(type(i))
           test_fl_y[h] = int(j)
        h += 1
   trainer.train_log_reg(train_sp_x,train_fl_y,test_sp_x,test_fl_y)
   print(train_sp_x.shape)
   
      
      
      
      
      
      
      
      