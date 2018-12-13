#-*- coding:utf-8 -*-
#########################################################################
# File Name: TenHello.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/python3

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Create  constanct op
#This iop is added a node to the default graph
hello = tf.constant("Hello, TensorFlow")

#start a TF session
sess = tf.Session()

#run the op and get result

print(sess.run(hello))
