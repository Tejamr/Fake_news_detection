from flask import Flask
from flask import render_template, request
import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

nltk.download('stopwords')

df = pd.read_csv("train.csv")
df = df.dropna()
X = df.drop('label', axis=1)
y = df['label']

def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', '', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

corpus = [preprocess_text(text) for text in X['title']]

voc_size = 5000
onehot_repr = [one_hot(words, voc_size) for words in corpus]

sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)


model = Sequential()
model.add(Embedding(voc_size, 40, input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(400))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.add(Dropout(0.5))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.load_weights("model_weights.h5") # model weights as .h5 file

###You can write Flask as seperate file if u need by just saving the weights and loading then in the .h5 file
#model.save('model.h5') # write this in the .py file i.e, for saving the weights in .h5 file
#model = load_model('model.h5') # write this in the Flask App file for loading the model weights


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_news():
    news_article = request.form['news_article']
    processed_article = preprocess_text(news_article)
    onehot_repr = [one_hot(processed_article, voc_size)]
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    prediction = model.predict_classes(embedded_docs)[0][0]
    result = 'Fake_news' if prediction == 1 else 'Real_News'

    return render_template('result.html', news_article=news_article, result=result)

if __name__ == '__main__':
    app.run(debug=True)
