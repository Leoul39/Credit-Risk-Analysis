import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("app/xg_model.pkl", "rb"))
scaler = pickle.load(open('app/scaler.pkl','rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def predict():
    prediction = None
    if request.method == 'POST':
        # Retrieve form data from each field
        provider_id = int(request.form['ProviderId'])                
        product_id = int(request.form['ProductId'])                  
        product_category = int(request.form['ProductCategory'])      
        channel_id = int(request.form['ChannelId'])
        amount = float(request.form['Amount'])                         
        recency = int(request.form['Recency'])                       
        frequency = int(request.form['Frequency'])                   
        features = np.array([provider_id,product_id,product_category,channel_id,amount,recency,frequency]).reshape(1,-1)
        #features[:, 4] = scaler.transform(features[:, 4].reshape(-1, 1)).flatten()
        prediction = model.predict(features)[0]
    print("Features array:", features)
    return render_template("index.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)