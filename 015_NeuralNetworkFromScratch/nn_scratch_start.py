
#%% packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%% data prep
# source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv('heart.csv')
df.head()

#%% separate independent / dependent features
X = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

#%% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#%% network class
class NeuralNetwork:
    def __init__(self, LR, X_train, y_train, X_test, y_test):
        self.w=np.random.randn(X_train.shape[1])
        self.b=np.random.randn()
        self.LR=LR
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.L_train=[]
        self.L_test=[]

    def activation(self,x):
        return 1/(1+np.exp(-x)) #Sigmoid function
    
    def dactivation(self,x):
        return self.activation(x)*(1-self.activation(x))
    
    def forward(self,X):
        hidden_1=np.dot(X,self.w)+self.b
        activate_1=self.activation(hidden_1)
        return activate_1
    
    def backward(self, X, y_true):
        hidden_1=np.dot(X,self.w)+self.b
        y_pred=self.forward(X)
        # Derivative of the loss function L=(y_pred-y_true)^2.
        dL_dpred=2*(y_pred-y_true)
        dpred_dh1=self.dactivation(hidden_1)
        dh1_db=1
        dh1_dw=X
        #Derivative of the loss function wrt w and b
        #in order to optimize the weights
        dL_db=dL_dpred*dpred_dh1*dh1_db
        dL_dw=dL_dpred*dpred_dh1*dh1_dw
        return dL_db, dL_dw
    
    def optimizer(self, dL_db,dL_dw):
        self.b=self.b- dL_db * self.LR
        self.w=self.w- dL_dw * self.LR
    
    def train(self,ITERATIONS):
        for i in range(ITERATIONS):
            # random position
            random_pos=np.random.randint(len(self.X_train))
            # forward path
            y_train_true=self.y_train[random_pos]
            y_train_pred=self.forward(X_train[random_pos])
            # calculate training losses
            L=np.square(y_train_pred-y_train_true)
            self.L_train.append(L)
            #Calculate gradients
            dL_db, dL_dw = self.backward(self.X_train[random_pos], self.y_train[random_pos])
            #update the weights
            self.optimizer(dL_db,dL_dw)
            #calulcate error for test data
            L_sum=0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])
                L_sum += np.square(y_pred - y_true)
            self.L_test.append(L_sum)
        return "Training succesful!"
        
#%% Hyper parameters
LR = 0.1
ITERATIONS = 1000
#%% model instance and training
nn = NeuralNetwork(LR=LR, X_train=X_train_scale, y_train=y_train,X_test=X_test_scale, y_test=y_test)
nn.train(ITERATIONS=ITERATIONS)
# %% check losses
sns.lineplot(x=list(range(len(nn.L_test))),y=nn.L_test)
# %% iterate over test data
total = X_test_scale.shape[0]
correct = 0
y_preds=[]
for i in range(total):
    y_true= y_test[i]
    y_pred = np.round(nn.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0

# %% Calculate Accuracy
correct/total
# %% In order to check if the model
# performs better than just simly using
# the class distribution to estimate the outcome,
# we will construct a baseline Classifier
from collections import Counter
Counter(y_test)
# %% Confusion Matrix
confusion_matrix(y_true=y_test, y_pred=y_preds)
# %%
