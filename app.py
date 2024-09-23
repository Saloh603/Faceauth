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

# Define threshold for comparison using VGG-Face
threshold = verification.find_threshold(model_name="VGG-Face", distance_metric="cosine")

# Path to the dataset
db_path = "deepface/tests/dataset"

@app.route('/verify', methods=['POST'])
def verify_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # Get the image from the request
    file = request.files['image']
    
    # Save the image temporarily to disk
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    try:
        # Find the face in the dataset
        dfs = DeepFace.find(img_path=img_path, db_path=db_path, silent=True)

        if len(dfs) == 0:
            return jsonify({"match": False}), 200
        
        # Check each DataFrame (there can be multiple for different faces)
        for df in dfs:
            if isinstance(df, pd.DataFrame):
                # Verify if the closest match distance is below the threshold
                if df["distance"].values[0] < threshold:
                    logger.info("✅ Face matched with distance below threshold.")
                    return jsonify({"match": True}), 200

        # If no match is found
        logger.info("❌ No match found.")
        return jsonify({"match": False}), 200
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up the temporary file
        if os.path.exists(img_path):
            os.remove(img_path)

# Start the Flask app
if __name__ == '__main__':
    # Make sure the 'uploads' folder exists to store temp images
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(host='0.0.0.0', port=9001, debug=True)