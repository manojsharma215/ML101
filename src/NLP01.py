import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I am a great catch!',
    'You are not a brave person',
    'Some people are more equal than others'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

tokenizer.fit_on_texts(sentences)

train_sequences = tokenizer.texts_to_sequences(sentences)

