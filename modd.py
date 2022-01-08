import numpy as np
from keras.preprocessing import image
from keras.models import load_model
model=load_model('static/classify-8_2.h5')
def modd(x):
    img=image.load_img(x,target_size=(100,100))
    img = img.resize((100, 100))
    img_tensor=image.img_to_array(img)
    img_tensor=np.expand_dims(img_tensor,axis=0)
    img_tensor/=255

    
    predict_x=model.predict(img_tensor) 
    a=np.argmax(predict_x,axis=1)
   
    return a

