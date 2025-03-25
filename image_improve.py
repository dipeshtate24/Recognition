import tensorflow as tf
import numpy as np
import imutils
import cv2
import os
from tensorflow.keras.preprocessing import image


def predict_meter_type(img_path):
    """
    Predicts the type of meter (Analog or Digital) from the given image.
    """
    model = tf.keras.models.load_model("Best_meter_recognition_model/meter_type_classification_best.h5")

    # Load image and preprocess
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict meter type
    prediction = model.predict(img_array)
    return "Analog Meter" if prediction[0][0] < 0.5 else "Digital Meter"


def preprocess_image(img_path, meter_type):
    """
    Preprocess the image based on meter type (digital or analog).
    Returns the processed image and the grayscale image.
    """
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image not found at {img_path}")

    if meter_type == "Analog Meter":
        # Enhance image details
        image = imutils.resize(image, height=100)

        # Enhance image details
        image = cv2.detailEnhance(image, sigma_s=20, sigma_r=0.15)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 70, 70)

        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        blur = cv2.GaussianBlur(gray, (5, 5), 2)

        # Adaptive + Otsu thresholding
        adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine both thresholding results
        thresh = cv2.bitwise_or(adaptive_thresh, otsu_thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Edge detection
        processed_img = cv2.Canny(thresh, 30, 150)

    else:

        image = imutils.resize(image, height=100)

        # Enhance image details
        image = cv2.detailEnhance(image, sigma_s=15, sigma_r=0.15)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.bilateralFilter(gray, 13, 70, 70)

        # Apply Gaussian Blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 1)

        # Adaptive + Otsu Thresholding for better segmentation
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine thresholding results
        thresh = cv2.bitwise_and(adaptive_thresh, otsu_thresh)

        # Morphological Transformations to connect broken segments
        kernel_close = np.ones((3, 3), np.uint8)  # Slightly larger kernel for better connection
        processed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return processed_img, gray


def split_image(img_path, save_dir):
    """Splits digits from meter image and saves them to the given directory."""
    meter_type = predict_meter_type(img_path)
    processed_img, gray = preprocess_image(img_path, meter_type)

    contours = cv2.findContours(processed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    digit_paths = []
    for i, c in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        if (meter_type == "Analog Meter" and 15 <= w <= 60 and 30 <= h <= 100) or (
                meter_type == "Digital Meter" and w >= 15 and 30 <= h <= 80):
            digit_img = gray[y:y + h, x:x + w]
            digit_resized = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
            digit_path = os.path.join(save_dir, f"digit_{i}.png")
            cv2.imwrite(digit_path, digit_resized)
            digit_paths.append(digit_path)
    return digit_paths


def digit_recognition(image_folder):
    """
    Loads a trained model and predicts digits from images in the given folder.
    """
    model_path = ("Best_digit_recognition_model/digit_recognition_model_CNN_4.h5")
    model = tf.keras.models.load_model(model_path)

    images = []
    file_paths = []

    # Sort filenames numerically
    file_names = sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))

    for file_name in file_names:
        img_path = os.path.join(image_folder, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        img_array = img.astype(np.float32) / 255.0
        img_array = img_array.reshape((28, 28, 1))
        images.append(img_array)
        file_paths.append(img_path)

    if not images:
        raise ValueError("No valid images found for digit recognition.")

    images_batch = np.array(images)
    predictions = model.predict(images_batch)
    recognized_digits = [np.argmax(prediction) for prediction in predictions]

    return recognized_digits
