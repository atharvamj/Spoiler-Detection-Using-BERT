from flask import Flask, render_template, request
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np

app = Flask(__name__)

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

# Custom BERT Model
inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
outputs = model(inputs).last_hidden_state
pooled_output = tf.reduce_mean(outputs, axis=1)
dense_layer = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(pooled_output)
dropout_layer = tf.keras.layers.Dropout(0.5)(dense_layer)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_layer)
model_custom_bert = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile BERT Model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model_custom_bert.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Load model weights
model_custom_bert.load_weights('C:/Users/mjdja/PycharmProjects/AML_1/disp/models/model_checkpoint-5.h5')

# Function to preprocess input text and make predictions
def predict_sentiment(text):
    custom_input_encodings = tokenizer(text, truncation=True, padding=True, return_tensors='tf')
    custom_predictions = model_custom_bert.predict(custom_input_encodings['input_ids'])
    print(custom_predictions)
    custom_predictions = (custom_predictions > 0.55).astype(int)
    print(custom_predictions)
    return(custom_predictions[[0]])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        prediction = predict_sentiment(user_input)
        result = "Spoiler Alert" if prediction == 1 else "No Spoilers here you are safe"
        return render_template('index.html', user_input=user_input, result=result)

if __name__ == '__main__':
    app.run(debug=True)
