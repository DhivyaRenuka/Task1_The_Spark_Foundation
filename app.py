import numpy as np
from flask import Flask , request , jsonify , render_template
import pickle
import os

app =Flask(__name__)
PEOPLE_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'study.jpg')
    return render_template('index.html', user_image =full_filename)

@app.route('/predict',methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'study.jpg')

    return render_template('index.html', predicted_value='Student Sore will be  {}'.format(output), user_image =full_filename)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[1]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
