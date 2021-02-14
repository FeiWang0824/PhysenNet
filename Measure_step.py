# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import Myfftshift

def AS(inpt,lamb,L,Z):
    
    M = int(inpt.shape[1]) 
    
    image = tf.cast(inpt,dtype = tf.complex128)
    image = 1j*image            
    U_in = tf.exp(image)             

    U_out=Myfftshift.ifftshift(tf.fft2d(Myfftshift.fftshift(U_in)))
    
    fx=1/L
    
    x = np.linspace(-M/2,M/2-1,M) 
    fx = fx*x                           
    [Fx,Fy]=np.meshgrid(fx,fx)
    
    k = 2*math.pi/lamb 
    H = tf.sqrt(1-lamb*lamb*(Fx*Fx+Fy*Fy))
    temp = k*Z*H
    temp = tf.cast(temp,dtype = tf.complex64)
    
    H = tf.exp(1j*temp)
    U_out = U_out*H
        
    U_out = Myfftshift.ifftshift(tf.ifft2d(Myfftshift.fftshift(U_out)))    
    I1 = tf.abs(U_out)*tf.abs(U_out)   
    I1 = I1/tf.reduce_max(tf.reduce_max(I1))
    
    return I1


