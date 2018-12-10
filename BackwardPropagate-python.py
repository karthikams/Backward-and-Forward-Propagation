
# coding: utf-8

# In[75]:


#Karthika Madhavanpillai Sasidharan Nair
import numpy as np
import pandas as pd

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

#input
X = np.array([[1,1,1], [1,0,1],[0,1,1],[0,0,1]])
y = np.array([[0],[1],[1],[0]])
print("y =")
print(y)


t = np.array([[0],[0],[0],[0]])

#input weights. Considering two nodes in one hidden layer
#Adding wo to the weight vector
w_hidden = np.random.random((4,5))

#weight for output layer 
#w_out = np.array([[1],[1]])
w_out = np.random.random((5,1))

#Adding bias x0 to input matrix
Xbias = np.array([[1,1,1,1], [1,0,1,1],[0,1,1,1],[0,0,1,1]])


epoch = 20000

for i in range(epoch):
    #product of inputs
    #z_hidden = Xbias.dot(w_hidden)
    z_hidden = np.dot(Xbias,w_hidden)

    #print("WX+B = ")
    #print(z_hidden)

    #Applying Activation Function(sigmoid)
    a_hidden = sigmoid(z_hidden)
    #print("Activation output")
    #print(a_hidden)

    k = np.array([[1],[1],[1],[1]])
    #a_hidden = np.append(a_hidden, k, axis = 1)
    # print("activation output after appending one column for bias")
    #print(activation_out)

    #final output
    #z_out = a_hidden.dot(w_out)
    z_out = np.dot(a_hidden, w_out)

    #print("z_out")
    #print(z_out)

    a_out =  sigmoid(z_out)
    #print("Sigmoid of out")
    #print(a_out)

    #validating whether a_out = y
    #print("a_out - y")
    #print(a_out - y)

    if np.array_equal(a_out-y,t):
        print("success")
        break;
    else:
        #print("fail")
        delta_a_out = a_out - y #(dE/da_out) 
        #print("delta aout ", delta_a_out )
        delta_z_out = sigmoid(a_out, True) #(da_out/dz_out)
        #print("delta zout ", delta_z_out )
        delta_wout = a_hidden      #(dzout/dwout)
        #print("delta wout ", delta_wout )
        delta_output_layer = np.dot(delta_wout.T,(delta_a_out*delta_z_out))
        #print("delta o/p layer ", delta_output_layer )



        #hidden layer
        delta_a_hidden = np.dot(delta_a_out * delta_z_out, w_out.T) #dE/dah = (dE/da0* da0/dzo * dzo/dah)
        #print("delta_a_hidden")
        #print(delta_a_hidden)
        delta_z_hidden = sigmoid(a_hidden, derive = True)           #dah/dzh =
        #print("delta_z_hidden")
        #print(delta_z_hidden)
        delta_w_hidden = Xbias                                      #dzh/dwh = Xbias

        delta_hidden_layer = np.dot(delta_w_hidden.T, delta_a_hidden*delta_z_hidden)
        #print("delta hidden")
        #print(delta_hidden_layer)

        w_hidden = w_hidden - delta_hidden_layer
        w_out = w_out - delta_output_layer
        
print("difference = ")       
print(a_out)
    

