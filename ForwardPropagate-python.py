
# coding: utf-8

# In[5]:


#Karthika Madhavanpillai
import numpy as np
import pandas as pd

#input
X = np.array([[1,1,1], [1,0,1],[0,1,1],[0,0,1]])
y = np.array([[0],[1],[1],[0]])

#adding bias also along with the input matrix
Xbias = np.array([[1,1,1,1], [1,1,0,1],[1,0,1,1],[1,0,0,1]])

#input weights. Considering two nodes in one hidden layer
wx = np.random.random((4,5))

#weight for hidden layer 1
#wout = np.matrix([[1],[1],[1]])
wout = np.random.random((5,1))

#product of inputs
result_hidden = Xbias.dot(wx)

#print("WX+B = ")
#print(result_hidden)

#Applying Activation Function(sigmoid)
activation_out = 1/(1+np.exp(-result_hidden))
#print("Activation output")
#print(activation_out)

k = np.array([[1],[1],[1],[1]])
#activation_out = np.append(activation_out, y, axis = 1)
#print("activation output after appending one column for bias")
#print(activation_out)

#final output
out_final = activation_out.dot(wout)

print("Final output")
print(out_final)

