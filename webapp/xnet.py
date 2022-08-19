import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json 
from tensorflow.keras.optimizers import Adam

def load_model(path:str='model'):
    # Routine to load the XNet model for inference
    # Reading the model architecture from the model json file
    json_file = open(os.path.join(path, 'XNet.json'),'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Use Keras model_from_json to restore the architecture
    loaded_model = model_from_json(loaded_model_json)
    print("\nSuccessfully restored model architecture!")

    # Load weights into the reconstructed model
    loaded_model.load_weights(os.path.join(path, 'XNet.h5'))
    print("Successfully restored model weights!")

    # Compile and return the loaded model
    optimizer = Adam(learning_rate=0.0001, decay=1e-5)
    loaded_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    print("Successfully compiled model!\n")

    return loaded_model

def predict(model, path:str='uploads'):
    # Transform each stock image into a 224x224 RGB image and 
    # then into a vector of the same size but normalized between 0 and 1
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:
        img = np.dstack([img, img, img])

    # Normalize the input image and convert into a tensor for inference
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    X = np.expand_dims(img, axis=0)
    X = np.array(X)
    X = tf.convert_to_tensor(X)

    # Run inference
    y = model.predict(X)
    file_name = os.path.basename(path)
    if y[0][0] > 0.5:
        result = file_name + ' is PNEUMONIA with {}% prediction confidence!'.format(round(y[0][0]*100))
    else:
        result = file_name + ' is NORMAL with {}% prediction confidence!'.format(round((1-y[0][0])*100))
    return result
