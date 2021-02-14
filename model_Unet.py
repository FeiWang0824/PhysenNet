# -*- coding: utf-8 -*-
"""
@author: Fei Wang
E-mail: WangFei_m@outlook.com
"""

import tensorflow as tf

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    
def inference(inpt, img_W, img_H, batch_size, isTrain=True):
        
    # Connect up all the downsampling layers.   
    inpt = tf.reshape(inpt,[batch_size,img_W,img_H,1])    
            
    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 32, 1])
        conv1 = tf.nn.conv2d_transpose(inpt,W_conv1,output_shape=[batch_size, img_W, img_H, 32],strides=[1,1,1,1],padding="SAME")   
        conv1 = tf.layers.batch_normalization(conv1, training = isTrain)    
        conv1 = tf.nn.relu(conv1)

        W_conv1_1 = weight_variable([3, 3, 32, 32])
        conv1_1 = tf.nn.conv2d(conv1, W_conv1_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv1_1 = tf.layers.batch_normalization(conv1_1, training = isTrain)    
        conv1_1 = tf.nn.leaky_relu(conv1_1)
        
    with tf.variable_scope('Max_Pooling_1'):
        Maxpool_1 = max_pool_2x2(conv1_1)
        
    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64])
        conv2 = tf.nn.conv2d(Maxpool_1, W_conv2, strides=[1, 1, 1, 1], padding="SAME")   
        conv2 = tf.layers.batch_normalization(conv2, training = isTrain)    
        conv2 = tf.nn.leaky_relu(conv2)

        W_conv2_1 = weight_variable([3, 3, 64, 64])
        conv2_1 = tf.nn.conv2d(conv2, W_conv2_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv2_1 = tf.layers.batch_normalization(conv2_1, training = isTrain)    
        conv2_1 = tf.nn.leaky_relu(conv2_1)
        
    with tf.variable_scope('Max_Pooling_2'):
        Maxpool_2 = max_pool_2x2(conv2_1)
    
    with tf.variable_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        conv3 = tf.nn.conv2d(Maxpool_2, W_conv3, strides=[1, 1, 1, 1], padding="SAME")   
        conv3 = tf.layers.batch_normalization(conv3, training = isTrain)    
        conv3 = tf.nn.relu(conv3)

        W_conv3_1 = weight_variable([3, 3, 128, 128])
        conv3_1 = tf.nn.conv2d(conv3, W_conv3_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv3_1 = tf.layers.batch_normalization(conv3_1, training = isTrain)    
        conv3_1 = tf.nn.leaky_relu(conv3_1)
         
    with tf.variable_scope('Max_Pooling_3'):
        Maxpool_3 = max_pool_2x2(conv3_1)
        
    with tf.variable_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 128, 256])
        conv4 = tf.nn.conv2d(Maxpool_3, W_conv4, strides=[1, 1, 1, 1], padding="SAME")   
        conv4 = tf.layers.batch_normalization(conv4, training = isTrain)    
        conv4 = tf.nn.leaky_relu(conv4)
        
        W_conv4_1 = weight_variable([3, 3, 256, 256])
        conv4_1 = tf.nn.conv2d(conv4, W_conv4_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv4_1 = tf.layers.batch_normalization(conv4_1, training = isTrain)    
        conv4_1 = tf.nn.leaky_relu(conv4_1)
         
    with tf.variable_scope('Max_Pooling_4'):
        Maxpool_4 = max_pool_2x2(conv4_1)
    
    with tf.variable_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 256, 512])
        conv5 = tf.nn.conv2d(Maxpool_4, W_conv5, strides=[1, 1, 1, 1], padding="SAME")   
        conv5 = tf.layers.batch_normalization(conv5, training = isTrain)    
        conv5 = tf.nn.leaky_relu(conv5)
        
        W_conv5_1 = weight_variable([3, 3, 512, 512])
        conv5_1 = tf.nn.conv2d(conv5, W_conv5_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv5_1 = tf.layers.batch_normalization(conv5_1, training = isTrain)    
        conv5_1 = tf.nn.leaky_relu(conv5_1)
         
    with tf.variable_scope('conv6'):
        W_conv6 = weight_variable([3, 3, 256, 512])
        conv6 = tf.nn.conv2d_transpose(conv5_1,W_conv6,output_shape=[batch_size, int(img_W/8), int(img_W/8), 256],strides=[1,2,2,1],padding="SAME")   
        conv6 = tf.layers.batch_normalization(conv6, training = isTrain)    
        conv6 = tf.nn.leaky_relu(conv6)
        
        merge1 = tf.concat([conv4_1,conv6], axis = 3)

        W_conv6_1 = weight_variable([3, 3, 512, 256])
        conv6_1 = tf.nn.conv2d(merge1, W_conv6_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv6_1 = tf.layers.batch_normalization(conv6_1, training = isTrain)    
        conv6_1 = tf.nn.leaky_relu(conv6_1)
        
        W_conv6_2 = weight_variable([3, 3, 256, 256])
        conv6_2 = tf.nn.conv2d(conv6_1, W_conv6_2, strides=[1, 1, 1, 1], padding="SAME")   
        conv6_2 = tf.layers.batch_normalization(conv6_2, training = isTrain)    
        conv6_2 = tf.nn.leaky_relu(conv6_2)
        
    with tf.variable_scope('conv7'):
        W_conv7 = weight_variable([3, 3, 128, 256])
        conv7 = tf.nn.conv2d_transpose(conv6_2,W_conv7,output_shape=[batch_size, int(img_W/4), int(img_H/4), 128],strides=[1,2,2,1],padding="SAME")   
        conv7 = tf.layers.batch_normalization(conv7, training = isTrain)    
        conv7 = tf.nn.leaky_relu(conv7)
        
        merge2 = tf.concat([conv3_1,conv7], axis = 3)

        W_conv7_1 = weight_variable([3, 3, 256, 128])
        conv7_1 = tf.nn.conv2d(merge2, W_conv7_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv7_1 = tf.layers.batch_normalization(conv7_1, training = isTrain)    
        conv7_1 = tf.nn.leaky_relu(conv7_1)
        
        W_conv7_2 = weight_variable([3, 3, 128, 128])
        conv7_2 = tf.nn.conv2d(conv7_1, W_conv7_2, strides=[1, 1, 1, 1], padding="SAME")   
        conv7_2 = tf.layers.batch_normalization(conv7_2, training = isTrain)    
        conv7_2 = tf.nn.leaky_relu(conv7_2)
        
    with tf.variable_scope('conv8'):
        W_conv8 = weight_variable([3, 3, 64, 128])
        conv8 = tf.nn.conv2d_transpose(conv7_2,W_conv8,output_shape=[batch_size, int(img_W/2), int(img_H/2), 64],strides=[1,2,2,1],padding="SAME")   
        conv8 = tf.layers.batch_normalization(conv8, training = isTrain)    
        conv8 = tf.nn.leaky_relu(conv8)
        
        merge3 = tf.concat([conv2_1,conv8], axis = 3)

        W_conv8_1 = weight_variable([3, 3, 128, 64])
        conv8_1 = tf.nn.conv2d(merge3, W_conv8_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv8_1 = tf.layers.batch_normalization(conv8_1, training = isTrain)    
        conv8_1 = tf.nn.leaky_relu(conv8_1)
        
        W_conv8_2 = weight_variable([3, 3, 64, 64])
        conv8_2 = tf.nn.conv2d(conv8_1, W_conv8_2, strides=[1, 1, 1, 1], padding="SAME")   
        conv8_2 = tf.layers.batch_normalization(conv8_2, training = isTrain)    
        conv8_2 = tf.nn.leaky_relu(conv8_2)
        
    with tf.variable_scope('conv9'):
        W_conv9 = weight_variable([3, 3, 32, 64])
        conv9 = tf.nn.conv2d_transpose(conv8_2,W_conv9,output_shape=[batch_size, img_W, img_H, 32],strides=[1,2,2,1],padding="SAME")   
        conv9 = tf.layers.batch_normalization(conv9, training = isTrain)    
        conv9 = tf.nn.leaky_relu(conv9)
        
        merge4 = tf.concat([conv1_1,conv9], axis = 3)

        W_conv9_1 = weight_variable([3, 3, 64, 32])
        conv9_1 = tf.nn.conv2d(merge4, W_conv9_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv9_1 = tf.layers.batch_normalization(conv9_1, training = isTrain)    
        conv9_1 = tf.nn.leaky_relu(conv9_1)
        
        W_conv9_2 = weight_variable([3, 3, 32, 32])
        conv9_2 = tf.nn.conv2d(conv9_1, W_conv9_2, strides=[1, 1, 1, 1], padding="SAME")   
        conv9_2 = tf.layers.batch_normalization(conv9_2, training = isTrain)    
        conv9_2 = tf.nn.leaky_relu(conv9_2)
                
    with tf.variable_scope('conv10'):
        W_conv10 = weight_variable([3, 3, 32, 1])
        conv10 = tf.nn.conv2d(conv9_2, W_conv10, strides=[1, 1, 1, 1], padding="SAME")
        conv10 = tf.layers.batch_normalization(conv10, training = isTrain)    
        conv10 = tf.nn.relu(conv10)
    
    with tf.variable_scope('output'):         
        Pha_out = tf.reshape(conv10,[batch_size,img_W,img_H,1])
        
    return Pha_out
        
        
        
        
        
        
        
    
