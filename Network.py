# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:07:19 2019

@author: kk
"""
import tensorflow as tf
import numpy as np
import function
import math
import os





class DQN:
    
    def __init__(self, name, learningrate,symbol,size):
        self.lr = learningrate
        self.name = name
        self.input = None
        self.target = None
        self.loss = None
        self.q = None
        self.probabilities = None
        self.symbol=symbol
        #self.size = size
        self.trainstep = None
        self.build(name,size,size)
    
    def build(self,name,input_size,output_size):
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32,shape=(None,input_size), name="inputs")
            self.target = tf.placeholder(tf.float32,shape=(None,output_size), name="target")
            size1 =  input_size*input_size*9 
            #l1 = tf.layers.dense(self.input,size1,tf.nn.relu)
            with tf.variable_scope("l1"):
                w1 = tf.Variable(tf.random_normal([input_size,size1]),name="w1")
                
                #w1 = tf.get_variable("w1", [self.input,size1],initializer=w,collections=c_names)
                b1 = tf.Variable(tf.constant(0.1,shape=[size1])) 
                #b1 = tf.get_variable("b1", [1,size1],initializer=b,collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.input,w1)+b1)
            with tf.variable_scope("q"):
                w2 = tf.Variable(tf.random_normal([size1,output_size]),name="w1")
                #w2 = tf.get_variable("w2", [size1,9],initializer=w,collections=c_names)
                b2 = tf.Variable(tf.constant(0.1,shape=[output_size])) 
                #b2 = tf.get_variable("b2", [1,9],initializer=b,collections=c_names)
                self.q = tf.matmul(l1,w2)+b2
            #self.q = tf.layers.dense(l1,9,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),name="q")
            self.probabilities = tf.nn.softmax(self.q,name="probabilities")
            mse = tf.reduce_mean(tf.squared_difference(self.q,self.target))
            
            
            self.trainstep = tf.train.RMSPropOptimizer(self.lr).minimize(mse,name="train")
            
       
    
class DQNAgent:  
    
    def __init__(self, name, rd, lr, t, symbol,size):
        self.rd = rd
        self.lr = lr
        self.t = t
        self.states = []
        self.history = []
        self.value = []
        self.next_max = []
        self.name = name
        self.size = size
        self.nn = DQN(name,lr,symbol,size*size)
        self.t = t
        self.symbol = symbol
        

        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        super().__init__()
        
        
        
    def get_target(self):
        target_value = []
        for i in range(len(self.history)):
            target = np.copy(self.value[i])
            target[self.history[i]] = self.rd*self.next_max[i]
            target_value.append(target)
        
        return target_value
    
    def get_probs(self, state_input):
        q,probs = self.sess.run([self.nn.q, self.nn.probabilities],feed_dict={self.nn.input: [state_input]})
        return probs[0],q[0]
    
    
    
    def move(self,state):
        self.states.append(state)
        state_input = function.convert_input(state)
        probs, q = self.get_probs(state_input)
        #q = np.copy(q)
        for i in range(len(state_input)):
            if state_input[i] != 0:
                probs[i] = -1
        
        choice = np.argmax(probs)
        
        if len(self.history) > 0 :
            self.next_max.append(q[choice])
        
        self.history.append(choice)
        self.value.append(q)
            
        coordinate = function.get_choice(choice,int(math.sqrt(state.size)))    
        state[coordinate[0]][coordinate[1]] = self.symbol 
        
        return state
    
    def fallback(self,state,s):
        self.next_max.append(s)
        
        
        if self.t:
            targets = self.get_target()
            
            
            state_input = [function.convert_input(x) for x in self.states]
            
            self.sess.run([self.nn.trainstep],feed_dict={self.nn.input:state_input, self.nn.target:targets})
            #print(self.sess.run([self.nn.]))
# =============================================================================
#         print("targets is ")
#         print(targets)
#         print("history is ")
#         print(self.history)
#         print("next_max is ")
#         print(self.next_max)
#         print("value is ")
#         print(self.value)
# =============================================================================
        #print(targets)
        
    def clean(self):
        self.states.clear()
        self.history.clear()
        self.next_max.clear()
        self.value.clear()