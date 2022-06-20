import nltk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from ML_Pipeline import Utils
import pickle
from textblob import Word
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords


# Preprocess text data
def cleaning(df, stop_words):
    df_tmp = df.copy(deep=True)
    df_tmp['content'] = df_tmp['content'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    # Replacing the special characters
    df_tmp['content'] = df_tmp['content'].str.replace("[^0-9a-zA-Z\s]+", '')

    # Replacing the digits/numbers
    # df_tmp['content'] = df_tmp['content'].str.replace('d', '')

    # Removing stop words
    df_tmp['content'] = df_tmp['content'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))

    # Lemmatization
    df_tmp['content'] = df_tmp['content'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))

    return df_tmp


# Tokenize text data
def tokenize(df, df_new, is_train):
    if is_train == 1:
        tokenizer = Tokenizer(num_words=Utils.input_length, split=' ')
        tokenizer.fit_on_texts(df_new['content'].values)
        # saving tokenizer
        with open('../output/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        x = tokenizer.texts_to_sequences(df['content'].values)
        x = pad_sequences(x, Utils.input_length)
        return x
    else:
        with open('../output/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        x = tokenizer.texts_to_sequences(df['content'].values)
        x = pad_sequences(x, Utils.input_length)
        return x


# Function to call dependent functions
def apply(path, is_train):
    print("Preprocessing started....")

    df = pd.read_csv(path)[["content", "score"]]
    print("Number of training examples: ", df.shape[0])
    y_data = pd.get_dummies(df['score'])

    print("Data loading completed....")

    stop_words = stopwords.words('english')
    df_new = cleaning(df, stop_words)

    x_data = tokenize(df, df_new, is_train)

    print("Preprocessing completed....")
    return x_data, y_data


# Get prediction given a review
def apply_prediction(review, ml_model):
    df = pd.DataFrame([review], columns=["content"])
    stop_words = stopwords.words('english')
    df_new = cleaning(df, stop_words)

    x_data = tokenize(df, df_new, is_train=0)
    prediction = list(ml_model.predict(x_data)[0])
    max_value = max(prediction)
    max_index = prediction.index(max_value)
    return max_index + 1
