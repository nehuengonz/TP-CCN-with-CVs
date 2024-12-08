import sys
import os
import shutil
import pandas as pd
import concurrent.futures  # Para manejar tiempos límite
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QProgressBar
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from pdf2image import convert_from_path
import hashlib  # Para calcular hashes

class CVClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Initialize model and feature extractor
        self.model = None
        self.feature_extractor = None
        self.init_feature_extractor()

    def init_feature_extractor(self):
        """
        Initialize the ResNet50 feature extractor.
        """
        base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    def initUI(self):
        self.setWindowTitle("CV Classifier")
        self.setGeometry(100, 100, 400, 300)

        # Layout
        layout = QVBoxLayout()

        # Select folder button
        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.folder_button)

        # Select model button
        self.model_button = QPushButton("Select Model")
        self.model_button.clicked.connect(self.select_model)
        layout.addWidget(self.model_button)

        # Process button
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self.process_files)
        layout.addWidget(self.process_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Status: Waiting")
        layout.addWidget(self.status_label)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.folder_path = ""
        self.model_path = ""

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path = folder
            self.status_label.setText(f"Selected Folder: {folder}")

    def select_model(self):
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "H5 Files (*.h5);;All Files (*)")
        if model_file:
            try:
                self.model = load_model(model_file)
                self.model_path = model_file
                self.status_label.setText(f"Model Loaded: {os.path.basename(model_file)}")
            except Exception as e:
                self.status_label.setText(f"Error loading model: {e}")

    def process_files(self):
        if not self.folder_path:
            self.status_label.setText("Please select a folder first.")
            return
        if not self.model:
            self.status_label.setText("Please load a model first.")
            return

        output_folder = os.path.join(self.folder_path, "Filtered_CVs")
        temp_image_folder = os.path.join(self.folder_path, "Temp_Images")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(temp_image_folder, exist_ok=True)

        files = [f for f in os.listdir(self.folder_path) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".pdf")]
        total_files = len(files)

        self.progress_bar.setValue(0)
        processed_files = 0

        filtered_files = []  # List to store filtered CVs

        # Define timeout duration (in seconds)
        timeout_duration = 30

        for file in files:
            file_path = os.path.join(self.folder_path, file)
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.handle_file, file_path, temp_image_folder, output_folder)
                    result = future.result(timeout=timeout_duration)

                    if result:  # If file was processed successfully
                        filtered_files.append(result)

                processed_files += 1
                self.progress_bar.setValue(int((processed_files / total_files) * 100))
            except concurrent.futures.TimeoutError:
                print(f"Timeout: Processing {file} took too long.")
            except Exception as e:
                print(f"Error processing {file}: {e}")

        # Remove duplicate files
        self.remove_duplicates(filtered_files)

        # Clean up temporary image folder
        shutil.rmtree(temp_image_folder, ignore_errors=True)

        self.progress_bar.setValue(100)  # Ensure progress bar reaches 100%
        self.status_label.setText(f"Processing Complete! Results saved in {output_folder}")

    def handle_file(self, file_path, temp_image_folder, output_folder):
        """
        Handles processing of a single file (PDF or image).
        """
        save_curriculum = False  # Flag to determine if the curriculum should be saved

        # Convert PDF to images if necessary
        if file_path.endswith(".pdf"):
            images = convert_from_path(file_path, dpi=300)
            for i, page in enumerate(images):
                if i >= 2:  # Process only pages 1 and 2
                    break

                image_path = os.path.join(temp_image_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_page_{i + 1}.jpg")
                page.save(image_path, "JPEG")

                if self.process_image(image_path):
                    save_curriculum = True
        else:
            if self.process_image(file_path):
                save_curriculum = True

        if save_curriculum:
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            shutil.copy(file_path, output_path)
            return output_path  # Return path to processed file

        return None

    def process_image(self, image_path):
        """
        Preprocesses an image, extracts features, and makes predictions.
        Returns True if the image passes the model's condition, otherwise False.
        """
        img = load_img(image_path, target_size=(224, 224))  # Match ResNet50's expected input size
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = img_array.reshape(1, *img_array.shape)

        # Extract features
        features = self.feature_extractor.predict(img_array)

        # Make prediction
        prediction = self.model.predict(features)
        return prediction[0][0] > 0.5  # Return True if condition is met

    def remove_duplicates(self, file_paths):
        """
        Removes duplicate files from the filtered CVs based on content.
        """
        file_hashes = {}
        duplicates = []

        for file_path in file_paths:
            try:
                file_hash = self.calculate_file_hash(file_path)
                if file_hash in file_hashes:
                    duplicates.append(file_path)
                else:
                    file_hashes[file_hash] = file_path
            except Exception as e:
                print(f"Error hashing file {file_path}: {e}")

        # Remove duplicates
        for duplicate in duplicates:
            try:
                os.remove(duplicate)
                print(f"Removed duplicate: {duplicate}")
            except Exception as e:
                print(f"Error removing duplicate {duplicate}: {e}")

    def calculate_file_hash(self, file_path):
        """
        Calculates a hash for the content of a file.
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CVClassifierApp()
    window.show()
    sys.exit(app.exec_())
