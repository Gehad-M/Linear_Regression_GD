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
