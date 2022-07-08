#import packages
import pandas as pd

import pandas as pd
df=pd.read_csv('outbreak_detect.csv')
#droping NaN values
df.dropna(inplace=True)
#LABEL ENCODING
from sklearn import preprocessing
LE=preprocessing.LabelEncoder()
df.Outbreak=LE.fit_transform(df.Outbreak)

#feature Engineering
df=df.drop('pf',axis=1)
df=df.drop('Positive',axis=1)
df=df.drop('Rainfall',axis=1)

#Loading of data
import numpy as np
x=np.array(df[['maxTemp','minTemp','avgHumidity']])
y=np.array(df[['Outbreak']])

#splitting of data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Standard Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)

#training the model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
#predition
y_pred=model.predict(sc.transform(x_test))
print(y_pred)
#pickling
import pickle
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print("Success Loaded")