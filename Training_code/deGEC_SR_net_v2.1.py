# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:59:33 2020

@author: Chang-Jen-Wang
"""
# In[1]: import functions
import tensorflow as tf
import numpy as np
import scipy.io as sc 
from  X_ini2 import Phase_ini
import math

# In[2] parameters setting
N = 100
M = 4*N
snrdb_train=np.array([20],dtype=np.float64)
snr_train = 10.0 ** (snrdb_train/10.0)  #train_SNR_linear

# bacth_size
batch_size= 50 

# Training samples
train_size = 100
epochs=1

# Iteration number
itermax=10

# Decentralization clusters (group_C = 1  mean GEC-SR-Net, group_C >1 mean deGEC-SR-Net)
group_C = 1

# Signal sparsity range 0~1
obj_rho = 0.5

M_c = int(M/group_C)
cluster_tab = np.zeros([group_C,2])

for ii in range(group_C):
    cluster_tab[ii,:] = [ii*M_c,(ii+1)*M_c]

train_repeat_size = 40000

snrdb_test = snrdb_train

weight_mat= r'....'

# In[3] Define functions

# 1) Define initial damping value (X_damping)
def Variable(shape):
    value =  np.array([[0.9000,    0.8100,    0.7290,    0.6561,    0.5905,    0.5314,    0.4783,    0.4305,    0.3874,    0.3487]]) # 0.9^{t}
    C = tf.constant_initializer(value)
    damping = tf.get_variable('damping', shape=shape, initializer = C,dtype='float64')  
    return damping

# 2) Define the observed function: y = abs (z + w)
def abq_y(y_input):
    abs_y = abs(y_input)
    return abs_y

# 3) Generate training data: Since the PR problem is special, the initial solution needs to be estimated.
# A function (X_ini) calculate the initial solution. Since I am deGEC-SR-net, group_C_in represents the data of different clusters
def generate_data_iid_test(B,M,M_c,N,SNR,group_C_in):  
    sigma2 = 10.**(-SNR/10.)
    # channel (complex)
    H_ = (np.random.randn(B,group_C_in,M_c,N)+1j*np.random.randn(B,group_C_in,M_c,N))/np.sqrt(2*M)
    

    # Sparse signal (complex)
    bernoulli = np.random.rand(B,1,N,1) > (1-obj_rho)
    Gauss = np.sqrt(1/(2*obj_rho))*(np.random.randn(B,1,N,1)+1j*np.random.randn(B,1,N,1))
    x1_ = bernoulli* Gauss # complex-sprase-gaussian
    
    z_ = np.matmul(H_,x1_)
    z_abs = np.mean(np.square(np.abs(np.reshape(z_,(B,-1)))),-1)
    #    SNR = E{|Ax|^2}/ (M*sigma2)  
    sigma2_eq = sigma2 * z_abs
    
    w=np.sqrt(1/2)*(np.random.randn(B,group_C_in,M_c,1)+1j*np.random.randn(B,group_C_in,M_c,1))*np.sqrt(sigma2_eq)
    y_ = z_+w
    
    # Notices: y_, H_, z_,x1_ are complex numbers  
    # Tensorflow can only be pure real numbers, real / image part is converted into real numbers separately
    x_real = np.concatenate((np.real(x1_),np.imag(x1_)),2)
    H_top = np.concatenate((np.real(H_),np.imag(H_)),2)
    H_low = np.concatenate((-np.imag(H_),np.real(H_)),2)
    H_real = np.concatenate((H_top,H_low),3)
    y_abs = abq_y(y_) 
    
    y_real = np.concatenate((np.real(y_),np.imag(y_)),2)
    z_real = np.concatenate((np.real(z_),np.imag(z_)),2)
      
    # Initional X Phase-Reterival
    x2_M_c = Phase_ini(y_,B,M,N,y_abs,H_)
    x2_out_real = tf.tile(tf.expand_dims(x2_M_c,1), multiples=[1, group_C_in,1])        

    # Due to the previous multiplication operation, we set the fourth dimension to 1,
    # and subsequent training does not require the final dimension. Use tf.squeeze to remove
    with tf.Session() as sess:
         x2_out = sess.run(x2_out_real)
         y_abs_out = sess.run(tf.squeeze(y_abs,3))
         y_real_out = sess.run(tf.squeeze(y_real,3))
         z_real_out = sess.run(tf.squeeze(z_real,3))
         x_complex_out = sess.run(tf.squeeze(x1_,3))
         x_real_out = sess.run(tf.squeeze(x_real,3))
                   
    return y_abs_out, H_real, x_real_out, x_complex_out, y_real_out, z_real_out , x2_out

# 4) average variance
def mean_var(varin): # varin = (Batchsize, Groups,Each group size)
    t0 = varin.shape[0] # Batchsize
    t1 = varin.shape[2] # Each group size
    varin_mean = tf.reduce_mean(varin,2) # (Batchsize, Groups)
    varin_mean1 = tf.expand_dims(varin_mean,-1) 
    varin_mean2 = tf.matmul(varin_mean1,tf.ones([t0,1,t1],dtype='float64')) # (Batchsize, Groups, Each group size)
    return varin_mean2

  
# 5) Run Conjugate complex multiplication based on real numbers to 
def complex_conj_mul_real(a,b,c): # a'*b => c = -1  , a*b => c=1
    real_part = tf.multiply(a,b) 
    a2 = tf.concat([c*a[:,N:2*N],a[:,0:N]],-1) 
    imag_part = tf.multiply(a2,b) 
    return  real_part, imag_part


# 6) Disambig phase rotation
def disambig1Drfft(xhat,x):
    real_part , imag_part= complex_conj_mul_real(xhat,x,-1)
    
    real_part_sum = tf.reduce_sum(real_part,1,keepdims=True)
    imag_part_sum = tf.reduce_sum(imag_part,1,keepdims=True)
    nor_abs = tf.sqrt(tf.square(real_part_sum)+tf.square(imag_part_sum))
    
    real_aa = tf.divide(real_part_sum,nor_abs)
    imag_aa = tf.divide(imag_part_sum,nor_abs)
    
    xout_real = tf.multiply(real_aa, xhat[:,0:N]) - tf.multiply(imag_aa, xhat[:,N:2*N])
    xout_imag = tf.multiply(real_aa, xhat[:,N:2*N]) + tf.multiply(imag_aa, xhat[:,0:N])
    
    x_out = tf.concat([xout_real,xout_imag],-1)
    return x_out

# 7) Define the range of the damp function
def Damp_linear(Input):
    Output =  tf.minimum(tf.maximum(Input,0),1)
    return Output
#-----------------------------------------------------------------------------#
# In[4] Define Modules and Ext (A,B,C)
    
# 1) Module A (Nonlinear measurements)
    
def abs_estimation(ob_y,Wvar,pri_z,pri_zvar):
    
    pri_z_abs = tf.sqrt(tf.square(pri_z[:,:,0:M_c]) + tf.square(pri_z[:,:,M_c:2*M_c]))
    
    pp_abs = tf.concat([pri_z_abs,pri_z_abs],2)/np.sqrt(2)
    yy_abs = tf.concat([ob_y,ob_y],2)/np.sqrt(2)

    B = tf.divide(2*tf.multiply(pp_abs,yy_abs),(Wvar+pri_zvar))
    
    I0 = tf.minimum( tf.divide(B,(tf.sqrt(tf.square(B)+4))), tf.divide(B,(0.5+tf.sqrt(tf.square(B)+0.25))) )
    
    y_sca = tf.divide(yy_abs,(1+tf.divide(Wvar,pri_zvar)))
    p_sca = tf.divide(pp_abs,(1+tf.divide(pri_zvar,Wvar)))

    zhat = tf.multiply(p_sca + tf.multiply(y_sca,I0), tf.divide(pri_z,pp_abs))
    
    C_constant = tf.divide(pri_zvar, 1 + tf.divide(pri_zvar,Wvar))
 
    zhat_var_c = tf.square(zhat[:,:,0:M_c]) + tf.square(zhat[:,:,M_c:2*M_c])
    zhat_var_r = tf.concat([zhat_var_c/2,zhat_var_c/2],2)
    
    zvar = tf.square(y_sca) + tf.square(p_sca) + tf.multiply((1 + tf.multiply(B,I0)), C_constant) - zhat_var_r
    
    return zhat, zvar

# 2) Module Bx (linear trasform Z => AX)
def linear_C2B_each_group(A,mux,varx,muz,varz):
    varx_matrix = tf.matrix_diag(varx)
    varz_matrix = tf.matrix_diag(varz)
    hat_varx = tf.matrix_inverse( varx_matrix +  tf.matmul(tf.matmul(A , varz_matrix,adjoint_a = True), A, adjoint_a = False))
    zz = tf.matmul(varx_matrix,tf.expand_dims(mux,-1), adjoint_a = False) + tf.matmul(tf.matmul(A , varz_matrix,adjoint_a = True), tf.expand_dims(muz,-1), adjoint_a = False)
    hat_mux = tf.matmul(hat_varx, zz, adjoint_a = False)
    hat_mux_out  = tf.squeeze(hat_mux,-1)
    hat_varx_mean_out = tf.matrix_diag_part(hat_varx)
    
    return hat_mux_out, hat_varx_mean_out


# 3) Module Bz (linear trasform AX => Z)
def linear_C2A_each_group(A,mux,varx,muz,varz):

    varx_matrix = tf.matrix_diag(varx)
    varz_matrix = tf.matrix_diag(varz)
    
    hat_varx = tf.matrix_inverse( varx_matrix +  tf.matmul(tf.matmul(A , varz_matrix,adjoint_a = True), A, adjoint_a = False))
    zz = tf.matmul(varx_matrix,tf.expand_dims(mux,-1), adjoint_a = False) + tf.matmul(tf.matmul(A , varz_matrix,adjoint_a = True), tf.expand_dims(muz,-1), adjoint_a = False)
    hat_mux = tf.matmul(hat_varx, zz, adjoint_a = False)
    hat_muz = tf.matmul(A,hat_mux,adjoint_a = False)
    hat_varz = tf.matmul(tf.matmul(A,hat_varx,adjoint_a = False), A, adjoint_b = True)
    hat_muz_out  = tf.squeeze(hat_muz,-1)
    hat_varz_mean_out = tf.matrix_diag_part(hat_varz)
  
    return hat_muz_out, hat_varz_mean_out
    


# 4) Fusion step combine information of each cluster
def MRC_combination(x_hat,v_hat):    
    v_MRC = 1 / tf.reduce_sum(v_hat,1, keepdims=True)
    x_MRC= v_MRC * tf.reduce_sum(x_hat*v_hat,1,keepdims=True)    
    return x_MRC, v_MRC


# 5) Define Module C function (X is Bernoulli gaussian distribution)
def Sparse_gaussian_signal(obj_mean,obj_var,obj_rho,rhat,rvar):
    xhat0 = tf.cast(obj_mean,tf.float64)
    xvar0 = tf.cast(obj_var,tf.float64)
    xrho0 = tf.cast(obj_rho,tf.float64)   
   # Compute posterior mean and variance
    a = tf.exp(  tf.divide(-tf.square(rhat),2*rvar) + tf.divide(tf.square(xhat0 - rhat), (2*(xvar0 + rvar))) )
    
   
    c = tf.divide(1, tf.sqrt(tf.multiply(tf.cast(2*math.pi,tf.float64),(rvar + xvar0))))
    
    Z = tf.multiply(tf.divide((1 - xrho0), tf.sqrt( tf.multiply(tf.cast(2*math.pi,tf.float64),rvar))), a ) + tf.multiply(xrho0,c)
    
    xhat_1 = tf.divide(  (xrho0*tf.multiply(c,(tf.multiply(xhat0,rvar) + tf.multiply(rhat,xvar0))))  , (rvar + xvar0) )
    
    xhat = tf.divide( xhat_1,Z)

    x2hat = tf.multiply((xrho0 * tf.divide(c,Z)),   tf.square(  tf.divide(tf.multiply(rhat,xvar0), (rvar + xvar0))) +  tf.divide(tf.multiply(rvar,xvar0),(rvar + xvar0) ) )
    xvar = x2hat - tf.square(xhat)
    
    return xhat, xvar 
# 6) Ext mean & variance
def Ext_part_var(E_new,E_old):
    E_information = tf.divide(E_new, 1- tf.multiply(E_new, E_old))
    return E_information

def Ext_part_mean(E_mean_new,E_var_new,E_mean_old,E_var_old,E_var):
    E_mean_information = tf.multiply(E_var,(tf.divide(E_mean_new,E_var_new) - tf.multiply(E_mean_old,E_var_old)))
    return E_mean_information
# In[5]: Define related training functions
    
# 1) Define save function for training parameters
def Save(weight_file):
    dict_name={}
    for varable in tf.trainable_variables():  
        dict_name[varable.name]=varable.eval()   
    sc.savemat(weight_file, dict_name)   
    
# 2) Define batch train
def Train_batch(sess):
    v_B_= np.zeros([batch_size,]) 
    x_B_= np.zeros([batch_size,]) 
    x_B2C_= np.zeros([batch_size,]) 
    v_B2C_= np.zeros([batch_size,]) 
    z_C2A_= np.zeros([batch_size,]) 
    v_C2A_= np.zeros([batch_size,]) 
    z_A2C_= np.zeros([batch_size,])
    v_A2C_= np.zeros([batch_size,])
    x_C2B_= np.zeros([batch_size,]) 
    v_C2B_ = np.zeros([batch_size,])
    _loss = list() 
    packet = train_repeat_size//train_size
    packet2 = train_size//batch_size
    
    # Generate training data
    batch_Y, batch_H, batch_X,x_complex_, batch_un_Y, batch_Z, batch_Xini = generate_data_iid_test(train_size,M,M_c,N,snrdb_train,group_C)
    
    # Repeat update training parameters with the same data set
    for offset in range(packet):    
        # Random selection in the same data set
        batch_index_list = (np.random.choice(train_size,train_size,replace=False)).reshape(packet2,batch_size)
        for offset2 in range(packet2):
            batch_index = batch_index_list[offset2]
            batch_Y_b = batch_Y[batch_index]
            batch_H_b = batch_H[batch_index]
            batch_X_b = batch_X[batch_index]
            batch_Z_b = batch_Z[batch_index]
            batch_Xini_b = batch_Xini[batch_index]
            
            _, b_loss,x_B_,v_B_,x_B2C_,v_B2C_,z_C2A_,v_C2A_,z_A2C_,v_A2C_,x_C2B_,v_C2B_,v_MRC_,x_MRC_,v_C_,z_C_,v_C_mean_,x_C_,z_A_,v_A_mean_,damping_\
            = sess.run([optimizer,cost,X_BB,v_B_mean,x_B2C,v_B2C,z_C2A,v_C2A,z_A2C,v_A2C,x_C2B,v_C2B,v_MRC,x_MRC,hat_varz_mean,hat_muz,hat_varx_mean,hat_mux,z_A,v_A_mean,damping],\
                   feed_dict={Y_: batch_Y_b, A_: batch_H_b, X_: batch_X_b, Z_: batch_Z_b, Xini_: batch_Xini_b})
        
            print("Packet %d Train loss: %.6f" % ((offset2+1, b_loss)))
            print("damping1:")
            print(damping_)
            
        _loss.append(b_loss)
            
    return _loss, x_B_, v_B_, x_B2C_, v_B2C_, z_C2A_, v_C2A_, z_A2C_, v_A2C_, x_C2B_, v_C2B_,damping_ 
# 3) Define train function
def Train():
    print("\nTraining ...") 
    saver = tf.train.Saver() 
    
    with tf.Session() as sess:    
        tf.global_variables_initializer().run()              
        weight_file=weight_mat+'\deGEC_SR_opt_weight_.mat'
      
        for i in range(epochs):
            train_loss, x_hat_train, v_B_,x_B2C_,v_B2C_,z_C2A_,v_C2A_,z_A2C_,v_A2C_,x_C2B_,v_C2B_,damping_ = Train_batch(sess)
            Save(weight_file) # training parameters save
            
        print("\nTraining is finished.")
        saver.save(sess,"./checkpoint_dir/MyModel")       
    return train_loss,damping_,v_B_,x_B2C_,v_B2C_,z_C2A_,v_C2A_,z_A2C_,v_A2C_,x_C2B_,v_C2B_    
  
# In[6]: DeGEC-SR-Net architecture
with tf.Graph().as_default():
    # Input setting (initial point is input after calculation outside the network)
    A_ = tf.placeholder(tf.float64,shape=[batch_size,group_C,2*M_c,2*N])
    X_ = tf.placeholder(tf.float64,shape=[batch_size,1,2*N])
    Y_ = tf.placeholder(tf.float64,shape=[batch_size,group_C,M_c])
    Z_ = tf.placeholder(tf.float64,shape=[batch_size,group_C,2*M_c])
    Xini_ =  tf.placeholder(tf.float64,shape=[batch_size,group_C,2*N])  
    sigma2 = 10.**(-snrdb_test/10.)/2
    
    # Training parameters
    with tf.variable_scope('damping'):
        damping = Variable((itermax,))
    
    # Initial setting  of deGEC-SR
    v_B2C = 2*tf.ones((batch_size,group_C,2*N), dtype='float64')
    x_B2C = Xini_
    v_A2C = 40*tf.ones((batch_size,group_C,2*M_c), dtype='float64')
    z_A2C = tf.squeeze(tf.matmul(A_,tf.expand_dims(x_B2C,-1)),-1)  
    v_C2A = 40*tf.ones((batch_size,group_C,2*M_c), dtype='float64')
    z_C2A = z_A2C
    v_C2B = v_B2C
    x_C2B =  x_B2C
    
    noise_var= sigma2   
    # Iteration Layers
    for t in range(itermax):      
       # Module A
       z_A, v_A = abs_estimation(Y_,noise_var,z_C2A,1/v_C2A)   
       v_A_mean = mean_var(v_A)
       
       # Ext
       v_A2C_new = Ext_part_var(v_A_mean,v_C2A)
       z_A2C_new = Ext_part_mean(z_A,v_A_mean,z_C2A,v_C2A,v_A2C_new)
           
       # damping
       v_A2C = (Damp_linear(damping[t]))*(v_A2C) + (1- Damp_linear(damping[t])) *(tf.divide(1,v_A2C_new))
       z_A2C = (Damp_linear(damping[t]))*(z_A2C) + (1- Damp_linear(damping[t])) *(z_A2C_new)          
       
       
       # Module Bx
       hat_mux , hat_varx_mean = linear_C2B_each_group(A_,x_B2C,v_B2C,z_A2C,v_A2C)
       v_C_mean1 = mean_var(hat_varx_mean)
       # Ext
       v_C2B_new = Ext_part_var(v_C_mean1,v_B2C)
       x_C2B_new = Ext_part_mean(hat_mux,v_C_mean1,x_B2C,v_B2C,v_C2B_new)
       
       v_C2B = (tf.divide(1,v_C2B_new))
       x_C2B = (x_C2B_new) 
  
       #  Fusion-part
       x_MRC, v_MRC = MRC_combination(x_C2B,v_C2B)
       
       # Module-C
       x_B,v_B = Sparse_gaussian_signal(0,(1/obj_rho)/2,obj_rho,x_MRC[:,0,:],v_MRC[:,0,:])
       v_B_mean = mean_var(tf.expand_dims(v_B,1))
       x_B =  tf.expand_dims(x_B,1)
       
       # 1 dimension extend to group_C to compute Ext
       v_B_mean_G = tf.tile(v_B_mean, multiples=[1, group_C,1])                   
       x_B_G = tf.tile(x_B, multiples=[1, group_C,1]) 
          
       # Ext
       v_B2C_new = Ext_part_var(v_B_mean_G,v_C2B)
       x_B2C_new = Ext_part_mean(x_B_G,v_B_mean_G,x_C2B,v_C2B,v_B2C_new)  

       
       # damping       
       v_B2C = Damp_linear(damping[t])*(v_B2C) + (1-Damp_linear(damping[t]))*(tf.divide(1,v_B2C_new))
       x_B2C = Damp_linear(damping[t])*(x_B2C) +(1-Damp_linear(damping[t]))*(x_B2C_new)          
        

       #  Module Bz  
       hat_muz , hat_varz_mean = linear_C2A_each_group(A_,x_B2C,v_B2C,z_A2C,v_A2C)
       v_zC_mean1 = mean_var(hat_varz_mean)
   
       # Ext
       v_C2A_new = Ext_part_var(v_zC_mean1,v_A2C)
       z_C2A_new = Ext_part_mean(hat_muz,v_zC_mean1,z_A2C,v_A2C,v_C2A_new)
       
       # damping
       v_C2A = (tf.divide(1,v_C2A_new))
       z_C2A = (z_C2A_new)  
  
       # Remove the ambiguity of each estimate
       X_BB = disambig1Drfft(tf.squeeze(x_B,1),tf.squeeze(X_,1))

       
       # Output finishing calculation cost function
       if t == 0:
           xout_iteration = X_BB
           xout_solution  =  tf.squeeze(X_,1)         
       if t > 0: 
           xout_iteration = tf.concat([xout_iteration,X_BB],0) 
           xout_solution  = tf.concat([xout_solution, tf.squeeze(X_,1)],0)
              
    cost = (tf.nn.l2_loss( xout_iteration - xout_solution)*2)/batch_size/itermax
    
    learning_rate=5e-2
    
    with tf.variable_scope('opt'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)    
    #Run DeGEC-SR-Net
    train_loss_,damping_,damping2_,v_B_,x_B2C_,v_B2C_,z_C2A_,v_C2A_,z_A2C_,v_A2C_,x_C2B_,v_C2B_=Train()
    