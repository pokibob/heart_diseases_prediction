import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import sklearn
from sklearn import preprocessing as per
from google.colab import drive
drive.mount('/content/drive')
#we are reading our data
df=pd.read_csv("/content/drive/MyDrive/heart.csv")
#First 5 rows of our data
df.head()
scaler=per.MinMaxScaler()
rescaleData=scaler.fit_transform(df)
rescaleData=pd.DataFrame(rescaleData,index=df.index,columns=df.columns)
print(rescaleData)
y=df.target.values
x=df.drop(['target'], axis=1)
x=(x-np.min(x)) / (np.max(x) - np.min(x)).values
#using the min-max normalization
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size =0.2, random_state=0)
accuracies = {}
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#from keras.layers import MaxoutDense
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import MaxPool1D
model=Sequential()
model.add(Dense(units =8, kernel_initializer ='uniform', activation = 'relu', input_dim =13))
model.add(Dense(units =8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units=1, kernel_initializer =  'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 10,epochs = 50)

#Evatuate model
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier

def build_classifier():
   classifier = Sequential()
   classifier.add(Dense(units =8,kernel_initializer ='uniform', activation ='relu',input_dim=13))
   classifier.add(Dense(units =8,kernel_initializer ='uniform', activation = 'relu'))
   classifier.add(Dense(units =1,kernel_initializer ='uniform', activation = 'sigmoid'))
   classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
   return classifier

classifier = KerasClassifier(build_fn =build_classifier, batch_size = 10, epochs = 10)
scores = cross_val_score(estimator = classifier,X = x_train, y = y_train, cv = 10, n_jobs = 1)

yhat_train = (model.predict(x_train)>0.5)
yhat_test = (model.predict(x_test)>0.5)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_train, yhat_train)
confusion_matrix = metrics.confusion_matrix(y_train, yhat_train)
classification = metrics.classification_report(y_train, yhat_train)
print()
print('===========================CNN Model Evaluation====================')
print()
print("Cross Validation Mean Score:" "\n", scores.mean())
print()
print("Model Acuracy: ""\n",accuracy)
print()
print("Conufosin matrix:" "\n",confusion_matrix)
y=df.target.values
x_data = df.drop(['target'], axis = 1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values


x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

 #transpose matrices

x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


#initialize
def initialize(dimension):

  weight = np.full((dimension, 1),0.01)
  bias = 0.0
  return weight,bias

def sigmoid(z):

  y_head = 1/(1+np.exp(-2))
  return y_head

def forwardBackward(weight,bias,x_train,y_train):
  #Forward

  y_head = sigmoid(np.dot(weight.T, x_train) + bias)
  loss = -(y_train*np.log(y_head) +(1-y_train)*np.log(1-y_head))
  cost = np.sum(loss) / x_train. shape[1]
  #Backward

  derivative_weight =np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
  derivative_bias =np.sum(y_head - y_train)/x_train.shape[1]
  gradients ={"Derivative Weight" : derivative_weight , "Derivative Bais:" : derivative_bias}
  return cost,gradients
  def update(weight,bias,x_train,y_train,learningRate, iteration):
  costlist = []
  index = []
  #for each iteration, update weight and bias valuea
  for i in range(iteration):
     cost, gradients = forwardBackward(weight,bias,x_train,y_train)
     weight = weight -learningRate * gradients["Derivative Weight"]
     bias = bias - learningRate * gradients["Derivative Bias"]

     costList.append(cost)
     index.append(1)
  parameters ={"weight" : weight, "bais": bias}
  print("Iteration: ",iteration)
  print( "cost:",cost)
  plt.plot(index,costList)
  plt.xlabel("Number of Iteration")
  plt.ylabel("Cost")
  plt.show()
  return parameters,gradients
  def update(weight,bias,x_train,y_train,learningRate, iteration):
  costList= []
  index = []
  #for each iteration, update weight and biasvalues
  for i in range(iteration):
    cost,gradients = forwardBackward(weight,bias,x_train,y_train)
    weight = weight - learningRate * gradients["Derivative Weight"]
    bias = bias - learningRate * gradients["Derivative Bias"]

    costList.append(cost)
    index.append(i)

  parameters = {"weight": weight, "bias": bias}

  print("iteration:" ,iteration)
  print ("cost:", cost)

  plt.plot(index,costList)
  plt.xlabel("Number fo iteration")
  plt.ylabel ("Cost")
  plt.show()

  return parameters,gradients
def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
  dimension =x_train.shape[0]
  weight,bias = initialize(dimension)

  parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)

  y_prediction = predict (parameters ["weight"],parameters ["bias"],x_test)

  print("Manuel Test Accuracy: {:.2f} %".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
  def predict(weight,bias,x_test):
  z=np.dot(weight.T,x_test) + bias
  y_head = sigmoid(z)

  y_prediction = np.zeros((1,x_test.shape[1]))
  for i in range(y_head, shape[1]):
    if y_head[0,i] <= 0.5:
      y_prediction[0,i] = 0
    else:
      Y_prediction[0,1] = 1
  return y_prediction
  logistic_regression(x_train,y_train,x_test,y_test,1,100)
  lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
acc = lr.score(x_test.T,y_test.T)*100

accuracies ['Logistic Regression'] = acc
print("Test Accuracy {:.2f}%".format(acc))
 # KNModel
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)
#n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict (x_test.T)
print ("{} NN scores:{:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train.T, y_train.T)

acc=svm.score(x_test.T,y_test.T)*100
accuracies['SVM'] = acc
print("Test-Accuracy of SVM Algorithm: (:.2f)%" ,format(acc))

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)

acc =nb.score(x_test.T,y_test.T)*100
accuracies ["Naive Bayes"] =acc
print("Accuracy to Naive Bayes: {:.2f}%".format (acc))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train.T, y_train.T)

acc = dtc.score(x_test.T, y_test.T)*100
accuracies ['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2F}%".format(acc))
from sklearn.ensemble import VotingClassifier
model2 = KNeighborsClassifier(n_neighbors= 3)
model3 = GaussianNB()
model4 = SVC(random_state = 1)

model21 = VotingClassifier(estimators=[('dt', model2), ('ga',model3), ('svc', model4)], voting= 'hard')
model21.fit(x_train.T,y_train.T)
acc= model21.score(x_test.T,y_test.T)*100
accuracies['voting classifier'] = acc
print("voting classifier Accuracy Score: {:.2f}%".format (acc))
colors = ["purple", "green", "orange", "magenta", "#CFC6OE", "#0FBBAE", "#0FBBAE"]
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(8,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.show ()
y_head_d1 = model21.predict(x_test.T)
from sklearn.metrics import confusion_matrix

cm_d1 = confusion_matrix(y_test,y_head_d1)


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes", fontsize=24)

plt.subplot(1,1,1)
plt.title("voting clasifier")
sns.heatmap(cm_d1,annot=True, fmt="d", cbar=False, annot_kws={"size": 24})
