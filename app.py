import requests
from flask import Flask, request, jsonify
import os
import cv2
import pandas as pd
from deepface import DeepFace
from deepface.modules import verification
from deepface.commons.logger import Logger

app = Flask(__name__)

# Logger instance
logger = Logger()

def verify(image1, image2):
    result = DeepFace.verify(image1, image2)
    return result

def get_image_path_userid(userId):
    url = f"http://82.97.243.112:8080/api/image/get-one?userId={userId}"
    response = requests.get(url)
    return f"/root/Prob24/ProjectFiles/upload_folder{response.json().get('uploadPath')}"

def get_image_path_imageid(imageId):
    url = f"http://82.97.243.112:8080/api/image/get-one-id?imageId={imageId}"
    response = requests.get(url)
    return f"/root/Prob24/ProjectFiles/upload_folder{response.json().get('uploadPath')}"


@app.route('/verify', methods=['POST'])
def verify_face():
    userId = request.json['userId']
    imageId = request.json['imageId']
    image1 = get_image_path_userid(userId)
    image2 = get_image_path_imageid(imageId)
    response = verify(image1, image2)
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001, debug=True)