import numpy as np
import pickle
from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/17cal',methods=['POST','GET'])
def make_predict():
    if request.method=='POST':

        new_vector = np.zeros(52)

        velocity = request.form['velocity']
        range1 = request.form['range']
        ar = request.form['ar']
        sabot = request.form['sabot']
        lpmass = request.form['Lpmass']
        sabot = float(sabot)
        lpmass = float(lpmass)
        velocity = int(velocity)
        range1 = int(range1)
        ar = int(ar)
        
        new_vector[velocity] = 1

        new_vector[range1] = 1

        #Min Max Scaler

        sabot_OD = (sabot-0.174500)/(0.179500 - 0.174500)

        Lp_mass = (lpmass-0.034300)/(0.109810 - 0.034300)


        new_vector[50] = sabot_OD

        new_vector[51] = Lp_mass

        knn_pkl = open("KNNRegressionModel17Cal.pkl","rb")
        model = pickle.load(knn_pkl)

        predict = np.array([new_vector])
        y_hat = model.predict(predict) 
        result = y_hat[0]

        if ar == 50:
            output = result
        else:
            output = ((3,358,139,325,440*(lpmass**5)) - (1,190,249,942,220.58*(lpmass**4)) + (168,731,332,886.914*(lpmass**3)) - (11,958,623,195.2566*(lpmass**2)) + (423,734,956.448111*(lpmass)) - 6,005,160.1262179)

        return render_template('result.html', output=output)

if __name__ == '__main__':
    
    app.run()          