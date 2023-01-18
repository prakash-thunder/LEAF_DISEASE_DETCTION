from flask import Flask,request,render_template,flash
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
# def predict_name(model, img):
#     class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
#     img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
#     img_array = tf.expand_dims(img_array, 0) # Create a batch

#     predictions = model.predict(img_array)

#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * (np.max(predictions[0])), 2)
    # return predicted_class, confidence
app=Flask(__name__)
# app.secret_key='khgdskfhk'
MODEL = tf.keras.models.load_model(r"C:\Users\praka\Visual Studio CODE\LEAF_DISEASE_DETECTION\models\1")
@app.route('/')
def home():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method=="POST":
        f=request.files['img']
        f.save(secure_filename(f.filename))
        a=f.filename
        print(a)
        Actual=a.split('.')[0]
        img1=cv2.imread(f.filename)
        img1=cv2.resize(img1,(256,256))
        class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        img_array = tf.keras.preprocessing.image.img_to_array(img1)
        img_array = tf.expand_dims(img_array, 0)
        pred=MODEL.predict(img_array)
        prediction=np.argmax(pred)
        pre=class_names[prediction]
        confidence = round(100 *(np.max(pred)), 2)
        print(pre)
        return render_template('index.html',data=pre,data2=confidence,data3=Actual)
    else:
        'something went wrong'
if __name__=="__main__":
    app.run(debug=True)