import pandas as pd
import numpy as np
import pickle
from flask import Flask
from flask import request
from flask import render_template
import flask

def conv_date(str1):
    words = word = str1.split('/')
    year = int(word[2])
    if year!=2012:
        return 'e'
    day = int(word[0])
    month = int(word[1])
    if month == 2:
        day = day + 31
    elif month == 3:
        day = day + 59
    elif month == 4:
        day = day + 90
    elif month == 5:
        day = day + 120
    elif month == 6:
        day = day + 151
    elif month == 7:
        day = day + 181
    elif month == 8:
        day = day + 212
    elif month == 9:
        day = day + 243
    elif month == 10:
        day = day + 273
    elif month == 11:
        day = day + 304
    elif month == 12:
        day = day + 334

    return day


weather_data = pd.read_csv("raw/weather_data.csv")

filename = "raw/RF_model_v1.sav"
model = pickle.load(open(filename, 'rb'))


app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/submit")
def predict():
    sday_f = request.args.get('sdate')
    stime_f = request.args.get('stime')

    sday_f = conv_date(sday_f)
    if sday_f == 'e':
        return '<h2>Weather Data Not Found</h2>'
    try:

        idx = weather_data.index[weather_data['sday']==sday_f]

        f1 = int(stime_f)
        f2 = weather_data.iloc[idx[0],:][2]
        f3 = weather_data.iloc[idx[0],:][3]
        f4 = weather_data.iloc[idx[0],:][4]

        feature_vector = [f1, f2, f3, f4]
        feature_vector = np.expand_dims(feature_vector, axis=0)

        delay_out = model.predict(feature_vector)
        delay_f = delay_out[0]
    except:
        delay_f = "Weather Data Not Found"

    return '<h2> Train Delay is '+str(delay_f)+'</h2>'

if __name__ == "__main__":
    app.run(debug=True)

'''

sday_f = input('Enter starting date of the journey')
stime_f = input('Enter starting time of the journey')

sday_f = conv_date(sday_f)
idx = weather_data.index[weather_data['sday']==sday_f]

f1 = int(stime_f)
f2 = weather_data.iloc[idx[0],:][2]
f3 = weather_data.iloc[idx[0],:][3]
f4 = weather_data.iloc[idx[0],:][4]

feature_vector = [f1, f2, f3, f4]
print(feature_vector)
feature_vector = np.expand_dims(feature_vector, axis=0)

delay_out = model.predict(feature_vector)
delay_f = delay_out[0]

print(delay_f)
'''
