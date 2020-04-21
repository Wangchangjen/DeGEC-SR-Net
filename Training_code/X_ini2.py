# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:21:43 2019

@author: adm
"""
import numpy as np
import tensorflow as tf
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator


def Phase_ini(g_y,B,M,N,y_abs,g_H):
    delta = M/N
    yy = np.reshape(np.reshape(y_abs,-1),[B,M])
    yy_ = yy**2
    ymean = np.mean(yy_,1, keepdims=True)
    yy1 = yy_/ymean 
    Ta = (yy1-1)/(yy1+np.sqrt(delta)-1)
    Tb = Ta*ymean
    g_2 = np.reshape(g_H,[B,M,N])  
    x2_M_c = np.zeros([B,2*N])
    
    
    for i in range(B):
        
        def fn(xin):
            return np.matmul((g_2[i,:,:].conj().T),Tb[i,:]*np.matmul(g_2[i,:,:],xin))/M
        
        AA = LinearOperator((N,N), matvec = fn)
        [eig_v, xin2] = eigs(AA,1, which = 'LR', maxiter = 1e3)
        abs2_y = tf.concat([yy[i], yy[i]],0)/np.sqrt(2)
        xin2r = np.vstack((np.real(xin2),np.imag(xin2)))
        Hr = np.vstack((np.hstack((np.real(g_2[i,:,:]),-np.imag(g_2[i,:,:]))),np.hstack((np.imag(g_2[i,:,:]),np.real(g_2[i,:,:])))))
        u11 = np.matmul(Hr,xin2r)
        u_var_c = np.sqrt(np.square(u11[0:M,:]) + np.square(u11[M:2*M,:]))/np.sqrt(2)
        u_var_r =  tf.concat([u_var_c,u_var_c],0)
        u = tf.multiply(u_var_r[:,0],abs2_y[:])
        l = tf.multiply(u_var_r[:,0],u_var_r[:,0])
        s = tf.divide(tf.norm(u,axis=0,ord=2),tf.norm(l,axis=0,ord=2))
        x2 = tf.multiply(xin2r,s)  
            
        for iteration in range(3):
                z_ob = tf.matmul(Hr,x2)
                zhat_var_c = tf.sqrt(tf.square(z_ob[0:M,:]) + tf.square(z_ob[M:2*M,:]))/np.sqrt(2)
                zhat_var_r = tf.concat([zhat_var_c,zhat_var_c],0)       
                z_abs = tf.expand_dims(tf.multiply(abs2_y[:], tf.divide(z_ob[:,0],zhat_var_r[:,0])),-1)
                x2 = tf.matrix_solve_ls(matrix =Hr ,rhs = z_abs)
        
        with tf.Session() as sess: 
                  x2_M_c[i,:] = sess.run(x2[:,0])
    
        
    return x2_M_c