import numpy as np
import copy


#initialize inputs for training
inputs_A=[[1,0,0],[0,1,0],[0,0,1]]
inputs_B=[[0,0,1],[0,1,0],[1,0,0]]
#initialize inputs for testing
inputs_A_test=[[1,1,0],[0,1,1],[0,0,0],[1,0,1],[1,1,1],[1,0,0]]

#create copy of inputs for preparation of weight matrix
inputs_A_bipo = np.array(copy.deepcopy(inputs_A))
inputs_B_bipo = np.array(copy.deepcopy(inputs_B))

#create bipolar version of inputs
for i in range(0,len(inputs_A_bipo)):
    for j in range(0,len(inputs_A_bipo)):
        if inputs_A_bipo[i,j]== 0:
            inputs_A_bipo[i,j] = -1
            
for i in range(0,len(inputs_B_bipo)):
    for j in range(0,len(inputs_B_bipo)):
        if inputs_B_bipo[i,j]== 0:
            inputs_B_bipo[i,j] = -1

#create weight matrix 
W=np.dot(inputs_A_bipo,inputs_B_bipo)
#create weight matrix transpose version
W_trans=np.transpose(W)

###start training###
for iterations in range (1):
    #To store final output
    output_final=[]
    
    #loop through each input pattern
    for i in range (0,len(inputs_A_test)):
        
        #To store output coming out of layer B
        output_B=[]

        #dot product of inputs pattern and weight matrix, output coming out of layer A
        output_A=np.dot(inputs_A_test[i],W)
    
        #apply performance rule
        for i in output_A:
            if i>0:
                j=1
            elif i<0:
                j=0
            else:
                j=i
            
            output_B.append(j)
            
        #dot product of output coming out of layer B 
        #and the transpose version of weight matrix
        output_C=np.dot(output_B, W_trans)
        
        #apply performance rule
        for i in output_C:
            if i>0:
                j=1
            elif i<0:
                j=0
            else:
                j=i
                
            output_final.append(j)
     
    #split final output into equal chunks        
    output_FINAL=[output_final[i:i + len(inputs_A)] for i in range(0, len(output_final), len(inputs_A))]
    
    #create a copy of final output for the next iteration
    inputs_A_test = copy.deepcopy(output_FINAL)

print(output_FINAL)


        
