# Training code for "Phase Retrieval with Learning Unfolded Expectation Consistent Signal Recovery Algorithm"
(c) 2020 Chang-Jen Wang and Chao-Kai Wen e-mail: dkman0988@gmail.com and chaokai.wen@mail.nsysu.edu.tw

--------------------------------------------------------------------------------------------------------------------------
# Information:
- deGEC-SR: Decentralized expectation consistent signal recovery

For phase retrieval, GEC-SR is good performance and deGEC-SR is good efficient algorithm. For details, please refer to 

C. J. Wang, C. K. Wen, S. H. Tsai, and S. Jin, "Decentralized Expectation Consistent Signal Recovery for Phase Retrieval", IEEE Transactions on Signal Processing, vol. 68, pp. 1484-1499, 2020.

However, the convergence iteration of GEC-SR and deGEC-SR are many. To reduce iteration number largely, we proposed deGEC-SR-Net to update the damp parameters, please refer to

C. J. Wang, C. K. Wen, S. H. Tsai, and S. Jin, "Phase Retrieval with Learning Unfolded Expectation Consistent Signal Recovery Algorithm", IEEE Signal Process. Letters, 2020, to appear.

We provide the traing codes that you can apply "deGEC-SR-Net" in the  phase retreival problem of different training environment.


# How to train "deGEC-SR-Net":

- Step 1. Install python and tensorflow
  
- Step 2. Add folders (i.e., X_ini2.py & deGEC_SR_net_v2.1.py ) to the same directory
  
- Step 3. In deGEC_SR_net_v2.1.py, find the line 32  

  You can select GEC-SR or deGEC-SR based on group_C. 
  - "group_C = 1" is GEC-SR. 
  
  - "group_C > 1" is deGEC-SR.
  
- Step 4. Now, you are ready to run the training code: deGEC_SR_net_v2.1.py
 
