import keras

top_words = 5000
input_length = 500


def save_model(model):
    model.save("../output/gru-model.h5")
    return True


def load_model(model_path):
    model = None
    try:
        model = keras.models.load_model(model_path)
    except:
        print("Please enter correct path")
        exit(0)

    return model
