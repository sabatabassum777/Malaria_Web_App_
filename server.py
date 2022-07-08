from flask import Flask,request,render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('outbreak_detect.csv')
app=Flask(__name__)
#Deserialization
model=pickle.load(open('model.pkl','rb'))

@app.route('/')  #using GET we send webpage to Client(browser)
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])#gets input data from client(browser) to Flask Server
def predict():
    list1=[]
    ct=0
    for x in request.form.values():
        print(x)
        if ct!=3:
            list1.append(x)
            ct=ct+1
    #features=[int(float(x)) for x in request.form.values()]
    #print(features)
    features=list1
    print(features)
    final=[np.array(features)]
    x=df.iloc[:,0:3].values
    sc = StandardScaler().fit(x)
    output=model.predict(sc.transform(final))
    print(output)
    print(output[0])
    if output[0]==0:
        return render_template('index.html',pred=f' No Malaria Outbreak ')
    elif output[0]==1:
        return render_template('index.html',pred=f'Malaria Outbreak ')
if __name__ =='__main__':
    app.run(debug=True)