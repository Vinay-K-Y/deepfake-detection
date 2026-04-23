import cv2
import numpy as np

class FaceExtractor:
    def __init__(self, target_size=(224, 224)):
        """
        Initializes the face extractor using Haar Cascade.

        Args:
            target_size (tuple): Output face size (width, height)
        """
        self.target_size = target_size

        # Load OpenCV's pre-trained Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if self.face_cascade.empty():
            raise RuntimeError("❌ Failed to load Haar Cascade model")

    def extract_face(self, image_path):
        """
        Detects and extracts the largest face from an image.

        Args:
            image_path (str): Path to image file

        Returns:
            np.ndarray or None: Cropped face image or None if no face found
        """
        # Read image
        img = cv2.imread(image_path)

        if img is None:
            print("❌ Error: Unable to read image")
            return None

        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            print("⚠️ No face detected")
            return None

        # Select the largest face (best practice)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        # Crop face
        face = img[y:y + h, x:x + w]

        # Resize to model input size
        face = cv2.resize(face, self.target_size)

        return face

    def extract_face_from_array(self, image_array):
        """
        Optional: Use this if you already have image in memory (no temp file needed)

        Args:
            image_array (np.ndarray): Image as numpy array (BGR or RGB)

        Returns:
            np.ndarray or None
        """
        if image_array is None:
            return None

        # Convert to BGR if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            img = image_array.copy()
        else:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, self.target_size)

        return face