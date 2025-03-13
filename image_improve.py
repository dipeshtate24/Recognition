import tensorflow as tf
import numpy as np
import imutils
import cv2
import os


def is_digital_meter(img):
    """Detects if the meter is digital based on line detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    return lines is not None  # Returns True if lines are detected, indicating a digital meter


def preprocess_image(img, digital=True):
    """Preprocess image based on meter type (digital or analog)."""
    detail = cv2.detailEnhance(img, sigma_s=20, sigma_r=0.15)
    gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if digital:
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5, 5), np.uint8)
    processed_img = cv2.erode(cv2.dilate(thresh, kernel, iterations=1), kernel, iterations=1)

    return processed_img, gray


def split_image(orig_img, save_dir):
    """Splits digits from meter image and saves them to the given directory."""
    img = cv2.imread(orig_img)
    if img is None:
        print("Error: Image not found or path is incorrect.")
        return []

    digital = is_digital_meter(img)
    processed_img, gray = preprocess_image(img, digital)

    cnts = cv2.findContours(processed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    digit_paths = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Ensure the save directory exists

    for i, c in enumerate(cnts, start=1):
        x, y, w, h = cv2.boundingRect(c)
        if w >= 15 and 30 <= h <= 100:
            digit_img = gray[y:y + h, x:x + w]
            digit_resized = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)

            digit_path = os.path.join(save_dir, f"digit_{i}.jpg")
            cv2.imwrite(digit_path, digit_resized)
            digit_paths.append(digit_path)

    return digit_paths


def digit_recognition(image_path):
    """
    Loads a trained model and predicts digits from the given image paths.

    Args:
        image_paths (list): List of file paths to images for digit recognition.

    Returns:
        List of recognized digits.
    """
    images = []

    for file_name in os.listdir(image_path):
        img_path = os.path.join(image_path, file_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array.reshape((28, 28, 1))  # Add channel dimension
            images.append(img_array)
        except FileNotFoundError:
            print(f"File not found: {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    images_batch = np.array(images)

    if images_batch.shape[1:] != (28, 28, 1):
        raise ValueError(f"Input images should be of shape (28, 28, 1), but got {images_batch.shape[1:]}")

    model_path = ('C:/Users/Dipesh/PycharmProjects/pythonProject2/latest_project/model'
                  '/digit_recognition_model_CNN_1.h5')

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(images_batch)
    recognized_digits = [np.argmax(prediction) for prediction in predictions]

    return recognized_digits
