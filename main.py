import numpy as np

#create sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


#input data
X = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])


#output data
Y = np.array([[0],
             [1],
             [1],
             [0]])

#makes randomization deterministic
np.random.seed(1)

#synapses
syn0 = 2*np.random.random((3,4)) - 1 #3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes)
syn1 = 2*np.random.random((4,1)) - 1 #4x1 matrix of weights (4 nodes x 1 output)

#training step
for j in range(160000):
    #dot product matrix multiplication step
    #prediction layers
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    #error rate
    l2_error = Y - l2

    #make sure it goes down every iteration
    if(j % 10000) == 0:
        print "Error" + str(np.mean(np.abs(l2_error)))

    #multiply error rate by result of sigmoid function (derivative of prediction from layer 2)
    l2_delta = l2_error * nonlin(l2, deriv=True)
    #backpropogation
    l1_error = l2_delta.dot(syn1.T)
    #sigmoid now used to get derivative of layer 1
    l1_delta = l1_error * nonlin(l1, deriv=True)

    #update weights to reduce error rate ("gradient descent")
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "Output after training"
print l2
print "That's incredible!"
