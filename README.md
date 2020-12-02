# Linear_Regression_GD
## Gehad Mohammed
# UnivariateData
```python
import numpy as np 
# Load univariateData
data = np.loadtxt('univariateData.txt', delimiter=",")
# Features
X = data[:,0]
# Adding the biase feature X0=1
X = np.column_stack((np.ones((len(X),1)), X)) # Add a column of ones to x for bias X0=1
print("X.shape",X.shape,"\n")
# Split our features data into train dataset(85%) and test dataset(15%)
X_train= X[0:82,:]
X_Test = X[82:,:]
print("X_Train",X_train,"\n","Shape_X_train",X_train.shape,"\n")
print("X_Test",X_Test,"\n","Shape_X_Test",X_Test.shape,"\n")
# Split our Label data into train dataset(85%) and test dataset(15%)
y_train= data[0:82,1]
y_Test = data[82:,1]
print("y_Train",y_train,"\n","Shape_y_train",y_train.shape,"\n")
print("y_Test",y_Test,"\n","Shape_y_Test",y_Test.shape,"\n")
m = len(y_train) # number of training examples
te= len(y_Test)
iterations = 1500 # gradient descent sitting
alpha = 0.01  # gradient descent sitting
```

    X.shape (97, 2) 
    
    X_Train [[ 1.      6.1101]
     [ 1.      5.5277]
     [ 1.      8.5186]
     [ 1.      7.0032]
     [ 1.      5.8598]
     [ 1.      8.3829]
     [ 1.      7.4764]
     [ 1.      8.5781]
     [ 1.      6.4862]
     [ 1.      5.0546]
     [ 1.      5.7107]
     [ 1.     14.164 ]
     [ 1.      5.734 ]
     [ 1.      8.4084]
     [ 1.      5.6407]
     [ 1.      5.3794]
     [ 1.      6.3654]
     [ 1.      5.1301]
     [ 1.      6.4296]
     [ 1.      7.0708]
     [ 1.      6.1891]
     [ 1.     20.27  ]
     [ 1.      5.4901]
     [ 1.      6.3261]
     [ 1.      5.5649]
     [ 1.     18.945 ]
     [ 1.     12.828 ]
     [ 1.     10.957 ]
     [ 1.     13.176 ]
     [ 1.     22.203 ]
     [ 1.      5.2524]
     [ 1.      6.5894]
     [ 1.      9.2482]
     [ 1.      5.8918]
     [ 1.      8.2111]
     [ 1.      7.9334]
     [ 1.      8.0959]
     [ 1.      5.6063]
     [ 1.     12.836 ]
     [ 1.      6.3534]
     [ 1.      5.4069]
     [ 1.      6.8825]
     [ 1.     11.708 ]
     [ 1.      5.7737]
     [ 1.      7.8247]
     [ 1.      7.0931]
     [ 1.      5.0702]
     [ 1.      5.8014]
     [ 1.     11.7   ]
     [ 1.      5.5416]
     [ 1.      7.5402]
     [ 1.      5.3077]
     [ 1.      7.4239]
     [ 1.      7.6031]
     [ 1.      6.3328]
     [ 1.      6.3589]
     [ 1.      6.2742]
     [ 1.      5.6397]
     [ 1.      9.3102]
     [ 1.      9.4536]
     [ 1.      8.8254]
     [ 1.      5.1793]
     [ 1.     21.279 ]
     [ 1.     14.908 ]
     [ 1.     18.959 ]
     [ 1.      7.2182]
     [ 1.      8.2951]
     [ 1.     10.236 ]
     [ 1.      5.4994]
     [ 1.     20.341 ]
     [ 1.     10.136 ]
     [ 1.      7.3345]
     [ 1.      6.0062]
     [ 1.      7.2259]
     [ 1.      5.0269]
     [ 1.      6.5479]
     [ 1.      7.5386]
     [ 1.      5.0365]
     [ 1.     10.274 ]
     [ 1.      5.1077]
     [ 1.      5.7292]
     [ 1.      5.1884]] 
     Shape_X_train (82, 2) 
    
    X_Test [[ 1.      6.3557]
     [ 1.      9.7687]
     [ 1.      6.5159]
     [ 1.      8.5172]
     [ 1.      9.1802]
     [ 1.      6.002 ]
     [ 1.      5.5204]
     [ 1.      5.0594]
     [ 1.      5.7077]
     [ 1.      7.6366]
     [ 1.      5.8707]
     [ 1.      5.3054]
     [ 1.      8.2934]
     [ 1.     13.394 ]
     [ 1.      5.4369]] 
     Shape_X_Test (15, 2) 
    
    y_Train [17.592    9.1302  13.662   11.854    6.8233  11.886    4.3483  12.
      6.5987   3.8166   3.2522  15.505    3.1551   7.2258   0.71618  3.5129
      5.3048   0.56077  3.6518   5.3893   3.1386  21.767    4.263    5.1875
      3.0825  22.638   13.501    7.0467  14.692   24.147   -1.22     5.9966
     12.134    1.8495   6.5426   4.5623   4.1164   3.3928  10.117    5.4974
      0.55657  3.9115   5.3854   2.4406   6.7318   1.0463   5.1337   1.844
      8.0043   1.0179   6.7504   1.8396   4.2885   4.9981   1.4233  -1.4211
      2.4756   4.6042   3.9624   5.4141   5.1694  -0.74279 17.929   12.054
     17.054    4.8852   5.7442   7.7754   1.0173  20.992    6.6799   4.0259
      1.2784   3.3411  -2.6807   0.29678  3.8845   5.7014   6.7526   2.0576
      0.47953  0.20421] 
     Shape_y_train (82,) 
    
    y_Test [0.67861 7.5435  5.3436  4.2415  6.7981  0.92695 0.152   2.8214  1.8451
     4.2959  7.2029  1.9869  0.14454 9.0551  0.61705] 
     Shape_y_Test (15,) 
    
    


```python

def computeCost(xtrain, ytrain, theta):
    J = 0
    s = np.power(( xtrain.dot(theta) - np.transpose([ytrain]) ), 2)
    J = (1.0/(2*len(xtrain))) * s.sum( axis = 0 )
    return J

def gradientDescent(xtrain, ytrain, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    theta = np.zeros((2, 1))
    for i in range(num_iters):
        theta = theta - alpha*(1.0/len(xtrain)) * np.transpose(xtrain).dot(xtrain.dot(theta) - np.transpose([ytrain]))
        J_history[i] = computeCost(xtrain, ytrain, theta)
    return theta, J_history



def LRgd_clf_fit(xtrain, ytrain):
    theta,J_history = gradientDescent(xtrain, ytrain, alpha, iterations)
    return(theta)


def LRgd_clf_predict(xtest):
    #X_padded = np.column_stack((np.ones((len(n), 1)), n))
    #print(X_padded.shape)
    prediction = np.array(xtest).dot(theta)
    return(prediction)


def performance(ytest,ypred):
    J = 0
    s = np.power((ypred - np.transpose([ytest])), 2)
    J = (1.0 / (2 * len(ytest)) * s.sum(axis=0))
    return (J)
```


```python
theta, J_history = gradientDescent(X_train, y_train, alpha, iterations)
print("Theta \n",theta,"\n Cost_fubction_history \n",J_history)
```

    Theta 
     [[-3.47777511]
     [ 1.1733358 ]] 
     Cost_fubction_history 
     [[6.59647878]
     [6.06637201]
     [6.05184445]
     ...
     [4.7043099 ]
     [4.7042925 ]
     [4.70427516]]
    


```python
cost_fun_result = computeCost(X_train, y_train, theta)
print("cost_fun_result \n",cost_fun_result)
```

    cost_fun_result 
     [4.70427516]
    


```python
# fitting the model to our trained data to get the weights of features (theta)
theta = LRgd_clf_fit(X_train, y_train)
# Use the resulted weights from our classifier to predict the value of label y for test dataset
ypred=LRgd_clf_predict(X_Test)
print("\n Y_Test_predict \n",ypred)

print("\n performance \n",performance(y_Test,ypred))
```

    
     Y_Test_predict 
     [[ 3.97959527]
     [ 7.98419037]
     [ 4.16756366]
     [ 6.51576061]
     [ 7.29368225]
     [ 3.56458639]
     [ 2.99950787]
     [ 2.45860006]
     [ 3.21927366]
     [ 5.4825211 ]
     [ 3.4105274 ]
     [ 2.74724067]
     [ 6.25316805]
     [12.23788466]
     [ 2.90153433]]
    
     performance 
     [3.46698804]
    


```python
#predicted_y for X_Train dataset
ypred_trained=LRgd_clf_predict(X_train)
print("\n Y_Trained_predict \n",ypred_trained)

print("\n performance \n",performance(y_train,ypred_trained))
```

    
     Y_Trained_predict 
     [[ 3.69142399]
     [ 3.00807322]
     [ 6.51740328]
     [ 4.7393302 ]
     [ 3.39773804]
     [ 6.35818161]
     [ 5.2945527 ]
     [ 6.58721676]
     [ 4.13271559]
     [ 2.45296805]
     [ 3.22279367]
     [13.14135323]
     [ 3.2501324 ]
     [ 6.38810167]
     [ 3.14066017]
     [ 2.83406752]
     [ 3.99097662]
     [ 2.5415549 ]
     [ 4.06630478]
     [ 4.8186477 ]
     [ 3.78411752]
     [20.30574165]
     [ 2.96395579]
     [ 3.94486453]
     [ 3.05172131]
     [18.75107171]
     [11.57377659]
     [ 9.3784653 ]
     [11.98209745]
     [22.57379976]
     [ 2.68505387]
     [ 4.25380384]
     [ 7.37346908]
     [ 3.43528479]
     [ 6.15660252]
     [ 5.83076717]
     [ 6.02143423]
     [ 3.10029741]
     [11.58316328]
     [ 3.97689659]
     [ 2.86633425]
     [ 4.59770857]
     [10.25964049]
     [ 3.29671383]
     [ 5.70322556]
     [ 4.84481309]
     [ 2.47127209]
     [ 3.32921523]
     [10.25025381]
     [ 3.02438259]
     [ 5.36941153]
     [ 2.74993934]
     [ 5.23295257]
     [ 5.44321435]
     [ 3.95272588]
     [ 3.98334994]
     [ 3.8839684 ]
     [ 3.13948683]
     [ 7.4462159 ]
     [ 7.61447226]
     [ 6.8773827 ]
     [ 2.59928303]
     [21.48963748]
     [14.01431507]
     [18.76749841]
     [ 4.9915974 ]
     [ 6.25516273]
     [ 8.53249019]
     [ 2.97486782]
     [20.38904849]
     [ 8.41515661]
     [ 5.12805635]
     [ 3.5695144 ]
     [ 5.00063208]
     [ 2.42046665]
     [ 4.20511041]
     [ 5.36753419]
     [ 2.43173067]
     [ 8.57707695]
     [ 2.51527218]
     [ 3.24450038]
     [ 2.60996038]]
    
     performance 
     [4.70427516]
    


```python

```
# MultivariateData
```python
import numpy as np 
# Load multivariateData
data = np.loadtxt('multivariateData.txt', delimiter=",")

# Features
Xdata = data[:,:2]

# Feature Normalization
def featureNormalize(Xdata):
    X_norm = Xdata
    mu    = np.zeros((1, Xdata.shape[1])) # mean (for each feature)
    s = np.zeros((1, Xdata.shape[1]))    # standard deviation (range) of each feature values, (max-minimum but in implementation we use in calculating it the standard deviation)
    for i in range(Xdata.shape[1]):
    	mu[:,i] = np.mean(Xdata[:,i])  # the mean for (first feature) , then in 2nd iteration (the mean for all data of the second feature)
    	s[:,i] = np.std(Xdata[:,i])   # the standard deviation (range which will be used in the devision for getting normalized data)
    	X_norm[:,i] = (Xdata[:,i] - float(mu[:,i]))/float(s[:,i])
    return X_norm, mu, s

# Xdata (Data after normalization)
Xdata , mu, s= featureNormalize(Xdata) #47row
print("Xdata \n",Xdata,"mean \n", mu,"standard_deviation \n",s,"\n")

# Adding the biase feature X0=1
Xdata = np.column_stack((np.ones((len(Xdata),1)), Xdata)) # Add a column of ones to x for bias X0=1
print("X.shape \n",Xdata.shape,"\n")

# Split our features data into train dataset(85%) and test dataset(15%)
X_train= Xdata[0:39,:]
X_Test = Xdata[39:,:]
print("X_Train",X_train,"\n","Shape_X_train",X_train.shape)
print("X_Test",X_Test,"\n","Shape_X_Test",X_Test.shape)

# Split our Label data into train dataset(85%) and test dataset(15%)
y_train= data[0:39,2]
y_Test = data[39:,2]
print("y_Train",y_train,"\n","Shape_y_train",y_train.shape)
print("y_Test",y_Test,"\n","Shape_y_Test",y_Test.shape)
m = len(y_train) # number of training examples
te= len(y_Test)

# Set up for gradient descent
# Choose alpha value
iterations = 1500 # gradient descent sitting
alpha = 0.01  # gradient descent sitting

```

    Xdata 
     [[ 1.31415422e-01 -2.26093368e-01]
     [-5.09640698e-01 -2.26093368e-01]
     [ 5.07908699e-01 -2.26093368e-01]
     [-7.43677059e-01 -1.55439190e+00]
     [ 1.27107075e+00  1.10220517e+00]
     [-1.99450507e-02  1.10220517e+00]
     [-5.93588523e-01 -2.26093368e-01]
     [-7.29685755e-01 -2.26093368e-01]
     [-7.89466782e-01 -2.26093368e-01]
     [-6.44465993e-01 -2.26093368e-01]
     [-7.71822042e-02  1.10220517e+00]
     [-8.65999486e-04 -2.26093368e-01]
     [-1.40779041e-01 -2.26093368e-01]
     [ 3.15099326e+00  2.43050370e+00]
     [-9.31923697e-01 -2.26093368e-01]
     [ 3.80715024e-01  1.10220517e+00]
     [-8.65782986e-01 -1.55439190e+00]
     [-9.72625673e-01 -2.26093368e-01]
     [ 7.73743478e-01  1.10220517e+00]
     [ 1.31050078e+00  1.10220517e+00]
     [-2.97227261e-01 -2.26093368e-01]
     [-1.43322915e-01 -1.55439190e+00]
     [-5.04552951e-01 -2.26093368e-01]
     [-4.91995958e-02  1.10220517e+00]
     [ 2.40309445e+00 -2.26093368e-01]
     [-1.14560907e+00 -2.26093368e-01]
     [-6.90255715e-01 -2.26093368e-01]
     [ 6.68172729e-01 -2.26093368e-01]
     [ 2.53521350e-01 -2.26093368e-01]
     [ 8.09357707e-01 -2.26093368e-01]
     [-2.05647815e-01 -1.55439190e+00]
     [-1.27280274e+00 -2.88269044e+00]
     [ 5.00114703e-02  1.10220517e+00]
     [ 1.44532608e+00 -2.26093368e-01]
     [-2.41262044e-01  1.10220517e+00]
     [-7.16966387e-01 -2.26093368e-01]
     [-9.68809863e-01 -2.26093368e-01]
     [ 1.67029651e-01  1.10220517e+00]
     [ 2.81647389e+00  1.10220517e+00]
     [ 2.05187753e-01  1.10220517e+00]
     [-4.28236746e-01 -1.55439190e+00]
     [ 3.01854946e-01 -2.26093368e-01]
     [ 7.20322135e-01  1.10220517e+00]
     [-1.01841540e+00 -2.26093368e-01]
     [-1.46104938e+00 -1.55439190e+00]
     [-1.89112638e-01  1.10220517e+00]
     [-1.01459959e+00 -2.26093368e-01]] mean 
     [[2000.68085106    3.17021277]] standard_deviation 
     [[7.86202619e+02 7.52842809e-01]] 
    
    X.shape 
     (47, 3) 
    
    X_Train [[ 1.00000000e+00  1.31415422e-01 -2.26093368e-01]
     [ 1.00000000e+00 -5.09640698e-01 -2.26093368e-01]
     [ 1.00000000e+00  5.07908699e-01 -2.26093368e-01]
     [ 1.00000000e+00 -7.43677059e-01 -1.55439190e+00]
     [ 1.00000000e+00  1.27107075e+00  1.10220517e+00]
     [ 1.00000000e+00 -1.99450507e-02  1.10220517e+00]
     [ 1.00000000e+00 -5.93588523e-01 -2.26093368e-01]
     [ 1.00000000e+00 -7.29685755e-01 -2.26093368e-01]
     [ 1.00000000e+00 -7.89466782e-01 -2.26093368e-01]
     [ 1.00000000e+00 -6.44465993e-01 -2.26093368e-01]
     [ 1.00000000e+00 -7.71822042e-02  1.10220517e+00]
     [ 1.00000000e+00 -8.65999486e-04 -2.26093368e-01]
     [ 1.00000000e+00 -1.40779041e-01 -2.26093368e-01]
     [ 1.00000000e+00  3.15099326e+00  2.43050370e+00]
     [ 1.00000000e+00 -9.31923697e-01 -2.26093368e-01]
     [ 1.00000000e+00  3.80715024e-01  1.10220517e+00]
     [ 1.00000000e+00 -8.65782986e-01 -1.55439190e+00]
     [ 1.00000000e+00 -9.72625673e-01 -2.26093368e-01]
     [ 1.00000000e+00  7.73743478e-01  1.10220517e+00]
     [ 1.00000000e+00  1.31050078e+00  1.10220517e+00]
     [ 1.00000000e+00 -2.97227261e-01 -2.26093368e-01]
     [ 1.00000000e+00 -1.43322915e-01 -1.55439190e+00]
     [ 1.00000000e+00 -5.04552951e-01 -2.26093368e-01]
     [ 1.00000000e+00 -4.91995958e-02  1.10220517e+00]
     [ 1.00000000e+00  2.40309445e+00 -2.26093368e-01]
     [ 1.00000000e+00 -1.14560907e+00 -2.26093368e-01]
     [ 1.00000000e+00 -6.90255715e-01 -2.26093368e-01]
     [ 1.00000000e+00  6.68172729e-01 -2.26093368e-01]
     [ 1.00000000e+00  2.53521350e-01 -2.26093368e-01]
     [ 1.00000000e+00  8.09357707e-01 -2.26093368e-01]
     [ 1.00000000e+00 -2.05647815e-01 -1.55439190e+00]
     [ 1.00000000e+00 -1.27280274e+00 -2.88269044e+00]
     [ 1.00000000e+00  5.00114703e-02  1.10220517e+00]
     [ 1.00000000e+00  1.44532608e+00 -2.26093368e-01]
     [ 1.00000000e+00 -2.41262044e-01  1.10220517e+00]
     [ 1.00000000e+00 -7.16966387e-01 -2.26093368e-01]
     [ 1.00000000e+00 -9.68809863e-01 -2.26093368e-01]
     [ 1.00000000e+00  1.67029651e-01  1.10220517e+00]
     [ 1.00000000e+00  2.81647389e+00  1.10220517e+00]] 
     Shape_X_train (39, 3)
    X_Test [[ 1.          0.20518775  1.10220517]
     [ 1.         -0.42823675 -1.5543919 ]
     [ 1.          0.30185495 -0.22609337]
     [ 1.          0.72032214  1.10220517]
     [ 1.         -1.0184154  -0.22609337]
     [ 1.         -1.46104938 -1.5543919 ]
     [ 1.         -0.18911264  1.10220517]
     [ 1.         -1.01459959 -0.22609337]] 
     Shape_X_Test (8, 3)
    y_Train [399900. 329900. 369000. 232000. 539900. 299900. 314900. 198999. 212000.
     242500. 239999. 347000. 329999. 699900. 259900. 449900. 299900. 199900.
     499998. 599000. 252900. 255000. 242900. 259900. 573900. 249900. 464500.
     469000. 475000. 299900. 349900. 169900. 314900. 579900. 285900. 249900.
     229900. 345000. 549000.] 
     Shape_y_train (39,)
    y_Test [287000. 368500. 329900. 314000. 299000. 179900. 299900. 239500.] 
     Shape_y_Test (8,)
    


```python

def computeCost(xtrain, ytrain, theta):
    J = 0
    s = np.power(( xtrain.dot(theta) - np.transpose([ytrain]) ), 2)
    J = (1.0/(2*len(xtrain)) * s.sum( axis = 0 ))
    return J


def gradientDescentMulti(xtrain, ytrain, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    theta = np.zeros((3, 1))
    for i in range(num_iters):
        theta = theta - alpha*(1.0/len(ytrain)) * np.transpose(xtrain).dot(xtrain.dot(theta) - np.transpose([ytrain]))
        J_history[i] = computeCost(xtrain, ytrain, theta)
        #print("J_history", theta)
    return theta , J_history


def LRgd_clf_fit(xtrain, ytrain):
    theta , cost = gradientDescentMulti(xtrain, ytrain, alpha, iterations)
    return(theta)


def LRgd_clf_predict(xtest):
    prediction = np.array(xtest).dot(theta)
    return(prediction)

def performance(ytest,ypred):
    J = 0
    s = np.power((ypred - np.transpose([ytest])), 2)
    J = (1.0 / (2 * len(ytest)) * s.sum(axis=0))
    return (J)


```


```python
theta, J_history = gradientDescentMulti(X_train, y_train, alpha, iterations)
print("Theta \n",theta,"\n \n Cost_fubction_history \n \n",J_history)
```

    Theta 
     [[342680.09681975]
     [109883.64254241]
     [   678.94046348]] 
     
     Cost_fubction_history 
     
     [[6.86657428e+10]
     [6.72244819e+10]
     [6.58157948e+10]
     ...
     [2.10390558e+09]
     [2.10390558e+09]
     [2.10390557e+09]]
    


```python
cost_fun_result = computeCost(X_train, y_train, theta)
print("cost_fun_result \n",cost_fun_result)
```

    cost_fun_result 
     [2.10390557e+09]
    


```python
# fitting the model to our trained data to get the weights of features (theta)
theta , j= LRgd_clf_fit(X_train, y_train)
print(theta)
# Use the resulted weights from our classifier to predict the value of label y for test dataset
ypred=LRgd_clf_predict(X_Test)
print("\n Y_Test_predict \n",ypred)

print("\n performance \n",performance(y_Test,ypred))
```

    [[342680.09681975]
     [109883.64254241]
     [   678.94046348]]
    
     Y_Test_predict 
     [[365975.20623843]
     [294568.54375196]
     [375695.51385738]
     [422580.04851284]
     [230619.3995837 ]
     [181079.32911784]
     [322648.04301604]
     [231038.69471166]]
    
     performance 
     [1.92851025e+09]
    


```python
#predicted_y for X_Train dataset
ypred_trained=LRgd_clf_predict(X_train)
print("\n Y_Trained_predict \n",ypred_trained)

print("\n performance \n",performance(y_train,ypred_trained))
```

    
     Y_Trained_predict 
     [[356966.99814189]
     [286525.41664485]
     [398337.45076714]
     [259906.81317405]
     [483098.31198153]
     [341236.79368887]
     [277300.92382976]
     [262346.0642659 ]
     [255777.10726122]
     [271710.32212365]
     [334947.36676949]
     [342431.43370599]
     [327057.27901418]
     [690572.88064513]
     [240123.4224841 ]
     [385262.78212453]
     [246489.36907938]
     [235650.94111921]
     [428450.18030426]
     [487431.02830377]
     [309866.17876788]
     [325875.9133062 ]
     [287084.47681546]
     [338022.19770786]
     [606587.36431991]
     [216642.89531842]
     [266678.78058814]
     [415947.8461414 ]
     [370384.44223657]
     [431461.76587587]
     [319027.42621621]
     [200862.71980779]
     [348923.87103478]
     [501344.2872023 ]
     [316917.67626728]
     [263743.71469243]
     [236070.23624717]
     [361782.25495884]
     [652912.83880476]]
    
     performance 
     [2.10390557e+09]
    


```python

```
