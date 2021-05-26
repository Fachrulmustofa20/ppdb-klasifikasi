import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    convert_features = [float(x) for x in request.form.values()]
    final_features = [np.array(convert_features)]
    prediksi = model.predict(final_features)
    
    output = prediksi[0]
    
    return render_template('index.html', prediction_text='{}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
