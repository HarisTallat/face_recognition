import os
import io
import cv2
import numpy as np
import face_recognition
from firebase_admin import credentials, initialize_app, storage
from PIL import Image
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"C:\Users\Haris Tallat\Desktop\Face_reg_api\pythonProject1\identitytrace-b335f-firebase-adminsdk-x9tri-b8d22c99ad.json")
initialize_app(cred, {
    'storageBucket': 'identitytrace-b335f.appspot.com'
})

# Load encoded images and IDs into memory
encodings, image_ids = [], []

def download_and_encode_image(blob):
    try:
        print(f"Downloading and processing image: {blob.name}")
        # Temporarily Download the file from the firebase
        image_data = blob.download_as_bytes()

        # Load image
        image = Image.open(io.BytesIO(image_data))
        # converting into red,green and blue color
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect face
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            #Encode face
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            image_id = os.path.splitext(blob.name.split('/')[-1])[0]
            print(f"Encoded image ID: {image_id}")
            return face_encoding, image_id
        print(f"No face found in image: {blob.name}")
    except Exception as e:
        print(f"Error processing image {blob.name}: {e}")
    return None, None

def get_images_from_firebase():
    print("Retrieving images from Firebase...")
    bucket = storage.bucket()
    #list_blobs is a method that lists files within the specified bucket.
    blobs = bucket.list_blobs(prefix='missing_person_images/')

    global encodings, image_ids
    encodings = []
    image_ids = []

    # Process each image sequentially
    for blob in blobs:
        encoding, image_id = download_and_encode_image(blob)
        if encoding is not None:
            encodings.append(encoding)
            image_ids.append(image_id)

    print(f"Retrieved and encoded {len(encodings)} images from Firebase.")

# Call the function to load images into memory
get_images_from_firebase()

@app.route('/check_image', methods=['POST'])
def check_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect and encode the uploaded face
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return jsonify({'message': 'No face detected in the uploaded image'}), 400

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]

    # Compare with known encodings
    matches = face_recognition.compare_faces(encodings, face_encoding)
    match_indices = [i for i, match in enumerate(matches) if match]

    if match_indices:
        matched_ids = [image_ids[i] for i in match_indices]
        return jsonify({'message': 'Match found', 'matched_ids': matched_ids}), 200
    else:
        return jsonify({'message': 'No match found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
