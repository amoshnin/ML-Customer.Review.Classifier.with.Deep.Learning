import subprocess

from ML_Pipeline import Train_Model
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Utils import load_model, save_model

#input 0 for training, 1 for prediction and 2 for deployement
val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))
if val == 0:
    x_train, y_train = apply("../input/review_data.csv", is_train=1)
    ml_model = Train_Model.fit(x_train, y_train)
    model_path = save_model(ml_model)
    print("Model saved in: ", "output/gru-model")
elif val == 1:
    model_path = "../output/gru-model.h5"
    # model_path = input("Enter full model path: ")
    ml_model = load_model(model_path)
    x_test, y_test = apply("../input/test_review_data.csv", is_train=0)
    print("Testing Accuracy: ", ml_model.evaluate(x_test, y_test)[1] * 100.0, "%")
else:
    # For prod deployment
    '''process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )'''

    # For dev deployment
    process = subprocess.Popen(['python', 'ML_Pipeline/deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )

    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)
