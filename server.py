import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

array_test = [7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653]

@app.route('/predict', methods = ['GET'])
def predict():
    x_test = np.array([array_test])
    prediction = model.predict(x_test.reshape(1,-1))
    return jsonify({'Predicción': list(prediction)})

if __name__=="__main__":
    model = joblib.load('./models/best_model_felicidad.pkl')
    app.run(port=8080)


