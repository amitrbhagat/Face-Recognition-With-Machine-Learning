import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request
import matplotlib.image as matimg


# UPLOAD_FOLDER = 'static/upload'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'upload')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def index():
    return render_template('index.html')


def app():
    return render_template('app.html')


def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) # save image into upload folder
        #get predictions
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)
        print("ML model predicted successfully") 

        #generate report
        

    return render_template('gender.html') #GET Request  