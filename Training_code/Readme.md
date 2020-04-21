# Training code for "Phase Retrieval with Learning Unfolded Expectation Consistent Signal Recovery Algorithm"
(c) 2020 Chang-Jen Wang and Chao-Kai Wen e-mail: dkman0988@gmail.com and chaokai.wen@mail.nsysu.edu.tw

--------------------------------------------------------------------------------------------------------------------------
# Parameter description for "deGEC_SR_net_v2.1.py":
Training code divided into 6 cells
- 1-th cell: Import functions (python, tensorflow, ...)

- 2-th cell: Parameters setting

  N: The dimension of transmitted X
 
  M: The dimension of measurement Y
 
  snrdb_train: SNR set (dB)
 
  train_size: The number of training samples (Each epoch)
 
  epochs: The number of epochs. For each epoch, the training samples are different.
 
  batch_size: The number of batch size
 
  itermax: The number of iteration
 
  group_C: Degree of dispersion (group_C = 1 => GEC-SR-Net, group_C > 1 => deGEC-SR-Net)
 
  train_repeat_size:  The number of update parameter times in the same data samples
 
  weight_mat: Path for storing parameters
 
- 3-th cell: Define functions

  Variable: Define initial learned damping value.
  
  abq_y: Define the observed function: y = |z + w|
  
  generate_data_iid_test: Generate training data
  
  mean_var: Average variance of posterior estimation
  
  complex_conj_mul_real: Run Conjugate complex multiplication based on real numbers
  
  disambig1Drfft: Disambig global phase rotation.
  
  Damp_linear: Define the range of the damp function.
  
- 4-th cell: Define Modules (A, Bx, Bz, C) and Ext

  abs_estimation: Module A (Nonlinear measurements)
  
  linear_C2B_each_group: Module Bx (linear trasform Z = AX)
  
  linear_C2A_each_group: Module Bz (linear trasform Z = AX)
  
  MRC_combination: Fusion step combine information of each cluster
  
  Sparse_gaussian_signal: Module C function (X is Bernoulli gaussian distribution)
  
  Ext_part_var, Ext_part_mean: Extrinsic mean and variance
  
- 5-th cell: Define related training functions
  
  Save: Define save function for training parameters
  
  Train_batch: Define batch train
  
  Train: Define train function
  
- 6-th cell: DeGEC-SR-Net architecture and run training
