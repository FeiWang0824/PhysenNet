# -*- coding: utf-8 -*-
"""
@author: Fei Wang
E-mail: WangFei_m@outlook.com
"""
import tensorflow as tf
import numpy as np
from PIL import Image  
import matplotlib.pyplot as plt
import os
import model_Unet
import Measure_step

tf.reset_default_graph()

# parameters
dim = 440
img_W = dim
img_H = dim
batch_size = 1

lamb = 632.8e-6                  # wavelength
pixelsize = 8e-3                 # pixel size
N = dim                          # num of pixels
L = N*pixelsize                  # length of the object and image plane

Steps = 5000                     # iteration steps
LR = 0.01                        # learning rate
Z = 22.3                         # diffraction distance mm
noise_level = 1.0/30

# create results save path
def mkdir(path): 
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
resut_save_path = '.\\results\\'
model_save_path = '.\\models\\'
mkdir(resut_save_path)
mkdir(model_save_path)
model_save_path =  model_save_path + 'exp_step_%d_lr_%f.ckpt'%(Steps,LR)   

# load raw data
diffraction_name = 'diff_1.tif'
measure_temp = np.array(plt.imread(diffraction_name)) 
measure_temp = measure_temp[280:280+dim,225:225+dim]  # crop the ROI
measure_temp = measure_temp/np.max(measure_temp)
            
# define the NN structure
with tf.variable_scope('input'):
    rand = tf.placeholder(shape=(dim,dim), dtype=tf.float32, name = 'noise')
    raw_measure = tf.constant(measure_temp, dtype=tf.float32, name = 'diffraction')
    inpt = raw_measure + rand    
with tf.variable_scope('inference'):
    out = model_Unet.inference(inpt, img_W, img_H, batch_size)
    out = tf.reshape(out,[dim,dim])
    
# physical model
out_measure = Measure_step.AS(out,lamb,L,Z)

loss_m = tf.losses.mean_squared_error(out_measure, raw_measure) # cost function (free from the labeled data)

optimizer = tf.train.AdamOptimizer(learning_rate=LR)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss_m)

# iteratively adjust the weights 
print('start optimization...')    
saver = tf.train.Saver()
with tf.Session() as sess:      
    sess.run(tf.global_variables_initializer())
            
    m_loss = np.zeros([Steps,1])
    temp_m_loss = []            
    for step in range(Steps):
        
        new_rand = np.random.uniform(0, noise_level, size=(dim,dim))   
        
        # check loss value
        if step % 100 == 0:                    
            loss_measure = sess.run(loss_m, feed_dict = {rand: new_rand})
            m_loss[step] = loss_measure
            print('step:%d  measure loss:%f'%(step, loss_measure))                       
        
        # visualization
        if step % 500 == 0:            
            pha_out = sess.run(out, feed_dict={rand: new_rand}).reshape(dim,dim)           
            out_measure_temp = sess.run(out_measure, feed_dict={rand: new_rand}).reshape(dim,dim)                               
            
            plt.subplot(131)
            plt.imshow(measure_temp)
            plt.title('raw diff')  
            plt.subplot(132)
            plt.imshow(out_measure_temp)  
            plt.title('estimated diff') 
            plt.subplot(133)
            plt.imshow(pha_out)
            plt.title('estimated phase') 
            plt.show()         
        
        # save the results with lower measure loss
        if step>0 and step%100==0:
            temp_m_loss.append(m_loss[step])
            if m_loss[step]<=np.min(temp_m_loss):
                min_saved_loss = m_loss[step]
                saver.save(sess, model_save_path)
                
                new_rand = np.random.uniform(0, 0.0, size=(dim,dim))
                pha_out = sess.run(out, feed_dict={rand: new_rand}).reshape(dim,dim)            
                pha_out = pha_out - np.min(pha_out)            
                pha_out = pha_out*255/np.pi  # rescale 0~pi to 0~255                        
                image_out = Image.fromarray(pha_out.astype('uint8')).convert('L')            
                image_out.save(resut_save_path+'pha_step_%d_lr_%f.bmp'%(step,LR))
                print('results saved!')
                                    
        sess.run([train_op], feed_dict = {rand: new_rand})
        
        
        
        