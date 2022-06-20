from flask import Flask, request

import Utils
# from ML_Pipeline.Preprocess import apply_prediction

app = Flask(__name__)

model_path = '../output/gru-model.h5'
ml_model = Utils.load_model(model_path)


@app.post("/get-review-score")
def get_image_class():
    from ML_Pipeline.Preprocess import apply_prediction
    data = request.get_json()
    review = data['review']
    prediction = apply_prediction(review, ml_model)
    output = {"Review Score": prediction}
    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
