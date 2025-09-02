from flask import Flask, render_template, request # type: ignore
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Model # type: ignore
import os

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length=39):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def generate_caption(img_path, model, tokenizer):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    feature = model.predict(img, verbose=0)
    y_pred = predict_caption(model_caption, feature, tokenizer, 39)
    return y_pred

model_caption = pickle.load(open('best.pkl', 'rb'))
model_img = VGG16()
model_img = Model(inputs=model_img.inputs, outputs=model_img.layers[-2].output)

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    img_path = os.path.join(upload_folder, file.filename)
    file.save(img_path)

    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    pred = generate_caption(img_path, model_img, tokenizer)
    # print("Generated Caption:", pred)  # Debugging output
    pred = pred.split(" ")[1:]
    pred.pop()
    pred = " ".join(pred)
    return render_template('after.html', data=pred,image_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
