
"""
Face Extractor for Deepfake Detection
Detects and crops faces from images using MediaPipe
"""
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

class FaceExtractor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def extract_face(self, image_path):
        """
        Detects and crops face from an image.
        Returns cropped face or None if no face found.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # Convert to RGB (mediapipe needs RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        results = self.face_detector.process(img_rgb)

        if not results.detections:
            return None  # No face found

        # Take the first detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Convert relative coords to pixel coords
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Add padding around face (20%)
        padding = int(max(bw, bh) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        bw = min(w - x, bw + 2 * padding)
        bh = min(h - y, bh + 2 * padding)

        # Crop and resize
        face = img_rgb[y:y+bh, x:x+bw]
        face_resized = cv2.resize(face, self.target_size)

        return face_resized

    def process_dataset(self, input_dir, output_dir, splits=['train', 'test', 'valid']):
        """
        Process all images in dataset
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        total_processed = 0
        total_skipped = 0

        for split in splits:
            for category in ['real', 'fake']:
                src = input_path / split / category
                
                if not src.exists():
                    print(f"⚠️  Skipping {split}/{category} - folder not found")
                    continue
                
                dst = output_path / split / category
                dst.mkdir(parents=True, exist_ok=True)

                # Get all images
                images = list(src.glob('*.jpg')) + list(src.glob('*.png')) + list(src.glob('*.jpeg'))
                
                if len(images) == 0:
                    print(f"⚠️  No images found in {split}/{category}")
                    continue
                
                print(f"\nProcessing {split}/{category}: {len(images)} images")

                for img_path in tqdm(images, desc=f"{split}/{category}"):
                    face = self.extract_face(img_path)

                    if face is not None:
                        # Save as JPEG
                        save_path = dst / f"{img_path.stem}.jpg"
                        cv2.imwrite(
                            str(save_path),
                            cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                        )
                        total_processed += 1
                    else:
                        total_skipped += 1

        print(f"\n{'='*50}")
        print(f"✅ Face extraction complete!")
        print(f"   Processed: {total_processed}")
        print(f"   Skipped  : {total_skipped} (no face detected)")
        print(f"{'='*50}")

if __name__ == "__main__":
    print("Face Extractor - Deepfake Detection")
    print("=" * 50)
    
    extractor = FaceExtractor(target_size=(224, 224))
    
    # Adjust these paths to match your folder structure
    INPUT_DIR = "real_vs_fake/real-vs-fake"
    OUTPUT_DIR = "processed_data"
    
    extractor.process_dataset(INPUT_DIR, OUTPUT_DIR)
