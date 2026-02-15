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
        print("predictions") 

        #generate report
        report = []
        for i, obj in enumerate(predictions):
            gray_image = obj['roi'] # grayscale image (array) 
            eigen_image = obj['eig_img'].reshape(100, 100) # eigen image (array)
            gender_name = obj['prediction_name']# name
            score = round(obj['score']*100,2) # Probability score

            #save grayscale and eigen in predict folder
            gray_image_name = f'gray_{i}.jpg'
            eigen_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}', gray_image, cmap='gray')
            matimg.imsave(f'./static/predict/{eigen_image_name}', eigen_image, cmap='gray')   

            # save report
            report.append([
                gray_image_name,
                eigen_image_name,
                gender_name,
                score
            ])

        return render_template('gender.html', fileupload = True, report=report) # POST Request


    return render_template('gender.html', fileupload = False) # GET Request