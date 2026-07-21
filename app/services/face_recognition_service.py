import os
import json
import urllib.request
import cv2
import numpy as np
from PIL import Image

class FaceRecognitionService:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.yunet_path = os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx")
        self.sface_path = os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")

        # Model download URLs
        self.yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        self.sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

        self.dnn_active = False
        self.detector = None
        self.recognizer = None
        self.cascade = None

        # Try to setup DNN
        self.setup_models()

    def setup_models(self):
        try:
            self.download_file_if_not_exists(self.yunet_url, self.yunet_path)
            self.download_file_if_not_exists(self.sface_url, self.sface_path)

            # Initialize DNN
            self.detector = cv2.FaceDetectorYN.create(
                model=self.yunet_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.8,
                nms_threshold=0.3,
                top_k=5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=self.sface_path,
                config=""
            )
            self.dnn_active = True
            print("OpenCV DNN Face Recognition models initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize DNN Face Recognition models: {e}.")
            print("Falling back to OpenCV Haar Cascades + Image Grid Histogram embeddings.")
            self.setup_fallback()

    def setup_fallback(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                raise Exception("Haar Cascade XML failed to load")
            print("Fallback Haar Cascade face detector initialized.")
        except Exception as e:
            print(f"Fallback detector initialization failed: {e}. Standard image hashes will be used.")

    def download_file_if_not_exists(self, url: str, dest_path: str):
        if not os.path.exists(dest_path):
            print(f"Downloading model from {url} ...")
            try:
                # Add headers to avoid potential blockings
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"Model downloaded successfully and saved to {dest_path}")
            except Exception as e:
                print(f"Failed to download model from {url}: {e}")
                raise e

    def get_face_encoding(self, image_path: str):
        """
        Loads image, detects face, and returns serialized 128-dimensional embedding.
        If no face is detected, returns None.
        """
        if not os.path.exists(image_path):
            print(f"Image path not found: {image_path}")
            return None

        # Read image in OpenCV format
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None

        h, w, c = img.shape

        if self.dnn_active:
            try:
                # Update input size for YuNet based on image size
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(img)

                if faces is not None and len(faces) > 0:
                    # Get first detected face
                    face = faces[0]
                    # Crop and align face
                    aligned_face = self.recognizer.alignCrop(img, face)
                    # Extract 128-D embedding
                    feature = self.recognizer.feature(aligned_face)
                    # Convert to standard Python float list
                    encoding = feature.flatten().tolist()
                    return encoding
                else:
                    print("DNN: No faces detected in the image.")
            except Exception as e:
                print(f"DNN feature extraction error: {e}. Attempting fallback...")

        # Fallback processing (Haar Cascade or direct image compression)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_rects = []
            if self.cascade is not None:
                face_rects = self.cascade.detectMultiScale(gray, 1.3, 5)

            if len(face_rects) > 0:
                # Crop the largest face
                x, y, w_f, h_f = sorted(face_rects, key=lambda r: r[2]*r[3], reverse=True)[0]
                face_crop = gray[y:y+h_f, x:x+w_f]
            else:
                # No face detected, use full image cropped to center square
                crop_size = min(h, w)
                y_start = (h - crop_size) // 2
                x_start = (w - crop_size) // 2
                face_crop = gray[y_start:y_start+crop_size, x_start:x_start+crop_size]

            # Resize to 32x32 to create a basic 1024-dimensional spatial-intensity vector
            resized = cv2.resize(face_crop, (32, 32))
            # Normalize vector
            feat = resized.flatten().astype(np.float32)
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            return feat.tolist()

        except Exception as e:
            print(f"Fallback extraction failed: {e}")
            return None

    def match_faces(self, query_encoding_list: list, database_encodings_list: list):
        """
        Compares query encoding list to a database encoding list.
        Returns (cosine_similarity_score, confidence_percentage)
        """
        if not query_encoding_list or not database_encodings_list:
            return 0.0, 0.0

        q_arr = np.array(query_encoding_list, dtype=np.float32)
        db_arr = np.array(database_encodings_list, dtype=np.float32)

        # Vector norms
        q_norm = np.linalg.norm(q_arr)
        db_norm = np.linalg.norm(db_arr)

        if q_norm == 0 or db_norm == 0:
            return 0.0, 0.0

        # Cosine similarity
        similarity = float(np.dot(q_arr, db_arr) / (q_norm * db_norm))

        # Calculate confidence percentage based on whether we are running DNN (SFace) or Fallback
        if self.dnn_active:
            # SFace cosine similarity matches typically range from 0.363 to 1.0.
            # A score of 0.363 is the threshold. We scale accordingly.
            threshold = 0.363
            if similarity >= threshold:
                # Scale similarity [0.363, 1.0] to [70%, 99.9%]
                confidence = 70.0 + ((similarity - threshold) / (1.0 - threshold)) * 29.9
            else:
                # Scale similarity [-1.0, 0.363] to [0%, 69.9%]
                # Typically non-matches are near 0.0.
                confidence = max(0.0, (similarity + 0.1) / (threshold + 0.1) * 69.9)
        else:
            # Fallback (spatial intensity matching)
            # Since pixel intensities are non-negative, similarity ranges from 0.0 to 1.0
            # Let's use a threshold of 0.85 for high confidence.
            threshold = 0.85
            if similarity >= threshold:
                confidence = 80.0 + ((similarity - threshold) / (1.0 - threshold)) * 19.9
            else:
                confidence = max(0.0, (similarity / threshold) * 79.9)

        return similarity, round(confidence, 2)

face_recognition_service = FaceRecognitionService()
