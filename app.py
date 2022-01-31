# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 21:38:18 2022

@author: Robin
"""

import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import predictionlib
from base64 import b64encode
from PIL import Image
import io
import os
from gtts import gTTS
import IPython

app=Flask(__name__)
model = load_model('./static/model_5.h5')
idx_word_dic = pickle.load(open('./static/idx_word_dic.pkl',"rb"))
tokenizer = pickle.load(open('./static/tokenizer.pkl', 'rb'))
max_length = pickle.load(open('./static/max_length.pkl', 'rb'))
captions_dic =  pickle.load(open("./static/captions_dic.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    f = request.files['file']
    content = f.read()
    res_image = b64encode(content).decode("utf-8")

    img = Image.open(io.BytesIO(content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    target_size = (224, 224)
    img = img.resize(target_size, Image.NEAREST)
#     print(max_length)
    caption = predictionlib.generate_captions(model, img, tokenizer, idx_word_dic, max_length)
    
    speech = gTTS(text = caption, lang = 'en', slow = False)
    print(type(speech))
    speech.save(r'./static/speech.mp3')
    
    
#     IPython.display.display(IPython.display.Audio('speech_1.mp3'))

    
    return render_template('results.html', result='Predicted Caption: {}'.format(caption),
                           res_image=res_image)
    
    
    


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5001)