from flask import Flask, render_template, request, session
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7]])
    pred = model.predict(arr)

    dp = pd.read_csv('FertilizerData.csv')
    nr = dp[dp['Crop'] == pred[0]]['N'].iloc[0]
    pr = dp[dp['Crop'] == pred[0]]['P'].iloc[0]
    kr = dp[dp['Crop'] == pred[0]]['K'].iloc[0]
    ph = dp[dp['Crop'] == pred[0]]['pH'].iloc[0]
    disease = dp[dp['Crop'] == pred[0]]['Disease'].iloc[0]
    prevention = dp[dp['Crop'] == pred[0]]['Prevention'].iloc[0]
    
    result = {
        "Prediction" : pred[0].upper(),
        "Nitrogen" : nr,
        "phosphorus": pr,
        "potassium" : kr,
        "PH" : ph,
        "Disease": disease,
        "Prevention": prevention
    }
    return render_template('after.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)

