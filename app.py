import pickle
from flask import Flask ,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

# Load the model
xgb_model = pickle.load(open('./titanic_survival/xgbmodel.pkl','rb'))
scaler = pickle.load(open('./titanic_survival/scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    if(data.get('Sex')=='Male'):
        data.update({'Sex':1})
    else:
        data.update({'Sex':0})
    

    a = ['B','C','D','A','F','E','T','Z','G']
    b = [1,2,3,4,5,6,7,8,9]

    for i in range(len(a)):
            if(data.get("Block")==a[i]):
                print('matched')
                data.update({"Block":b[i]})
                break
            else:
                pass
            
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.fit_transform(np.array(list(data.values())).reshape(1,-1))
    output = xgb_model.predict(new_data)
    print(output[0])

    return jsonify(int(output[0]))



@app.route('/predict',methods=['POST'])
def predict():
    # data=[float(x) for x in request.form.values()]
    data = list(request.form.values())

    for i in range(len(data)):
        if 'Male' in data:
            index = data.index('Male')
            data[index] = 1
        if 'Female' in data:
            index = data.index('Female')
            data[index] = 0
    

    a = ['B','C','D','A','F','E','T','Z','G']
    b = [1,2,3,4,5,6,7,8,9]


  
    for i in range(len(a)):
            if a[i] in data:
                index = data.index(a[i])
                data[index]=b[i]
                print('matched')
                break
            else:
                pass
    
    print(data)
    final_input = scaler.fit_transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = xgb_model.predict(final_input)[0]
    # survival = 'Survived' if output==1 else 'Not survived'
    return render_template('home.html',prediction_no='The prediction of survival is {}'.format(output))
    # return data

if __name__=="__main__":
    app.run(debug=True)

