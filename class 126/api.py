from flask import Flask,jsonify, request
from predict import getPrediction

app = Flask(__name__)

@app.route("/predictdata" , methods =["POST"])
def prediction():
    img = request.files.get("digit")
    pred = getPrediction(img)
    return jsonify({
        "prediction" : pred
    }),200

if (__name__ == "__main__"):
    app.run(debug=True)
