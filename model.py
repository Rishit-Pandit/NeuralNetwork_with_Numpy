import numpy as np

# activation functions
# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of sigmoid function
def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))

# trining function
def train(x, y, iterations):
    #random init of weights
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()
    
    learning_rate = 0.1
    costs = [] # keep costs during training, see if they go down   
    for i in range(iterations):
        # get a random point
        ri = np.random.randint(len(x))
        pointx = x[ri]
        pointy = y[ri]
        
        z = pointx[0] * w1 + pointx[1] * w2 + b
        pred = sigmoid(z) # networks prediction
        
        target = pointy
        
        # cost for current random point
        cost = np.square(pred - target)
        
        # print the cost over all data points every 1k iters
        if i % 100 == 0:
            c = 0
            for j in range(len(x)):
                px = x[j]
                py = y[j]
                p_pred = sigmoid(w1 * px[0] + w2 * px[1] + b)
                c += np.square(p_pred - py)
            costs.append(c)
        
        dcost_dpred = 2 * (pred - target)
        dpred_dz = sigmoid_p(z)
        
        dz_dw1 = pointx[0]
        dz_dw2 = pointx[1]
        dz_db = 1
        
        dcost_dz = dcost_dpred * dpred_dz
        
        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db
        
        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db
        
    return w1, w2, b

# prediction function
def predict(w_1, w_2, B, x, y): 
    z = w_1 * x[0] + w_2 * x[1] + B
    pred = float(sigmoid(z))
    real = y
    error = real - pred
    return pred, error, real

# the dataset
x = [[3, 1.5],[2, 1],[4, 1.5],[3, 1],[3.5, 0.5],[2, 0.5],[5.5, 1],[1, 1]]
y = [[1], [0], [1], [0], [1], [0], [1], [0]]

# Implimenting all of it
w1, w2, b = train(x, y, 1000000)
pred, error, real = predict(w1, w2, b, [4.5, 1], 1)

# printing the information
print('pred: ', pred)
print('real: ', real)
print('error: ', error)
