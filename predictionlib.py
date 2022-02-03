import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model, Model


def image_encodings(img):
    
    model = VGG16(include_top=True, weights='imagenet')
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    img_encoding = model.predict(img, verbose = 0)
    
    return img_encoding

def generate_captions(model, img, tokenizer, idx_word_dic, max_len):
    
    caption = 'sos'
    img_encoding = image_encodings(img)
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], max_len)
        yhat = model.predict([img_encoding, seq])
        yhat = np.argmax(yhat)
        pred_word = idx_word_dic[yhat]
        if pred_word == None:
            break
        caption += " " + pred_word
        if pred_word == 'eos':
            break
    
    caption = ' '.join((caption.split()[1:-1]))
        
    return caption
    
