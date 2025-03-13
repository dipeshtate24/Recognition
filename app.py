import cv2
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from image_improve import split_image
from image_improve import digit_recognition
import image_crop as crop
import uuid
import os

app = Flask(__name__)

display_folder = 'static'

img_crop_folder = 'img_crop'

split_digit_folder = 'split_digit'

allowed_extensions = {'jpeg', 'jpg', 'png'}


if not os.path.exists(display_folder):
    os.makedirs(display_folder)  # Create the upload folder if it doesn't exist

if not os.path.exists(img_crop_folder):
    os.makedirs(img_crop_folder)  # Create the img_crop folder if it doesn't exist

if not os.path.exists(split_digit_folder):
    os.makedirs(split_digit_folder)  # Create the split_digit folder if it doesn't exist


def allowed_type(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


file_urls = []


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global file_urls
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']

            if image.filename == '':
                flash("Image name is not valid", "danger")
                return redirect(request.url)

            if not allowed_type(image.filename):
                flash("That image extension is not allowed", 'danger')
                return redirect(request.url)

            filename = secure_filename(image.filename)
            image_path = os.path.abspath(display_folder)
            crop_path = os.path.abspath(img_crop_folder)

            try:
                # Save uploaded image
                image.save(os.path.join(image_path, filename))

                # Crop image
                crop_image = crop.img_crop(os.path.join(image_path, filename))

                if crop_image is None:
                    flash("Error processing image", "danger")
                    return redirect(request.url)

                # Save cropped image with a unique name
                unique_name = f"{uuid.uuid4().hex}.jpg"
                crop_save_path = os.path.join(crop_path, unique_name)
                cv2.imwrite(crop_save_path, crop_image)

                # Get a list of all cropped images
                crop_img_list = sorted(os.listdir(img_crop_folder))
                file_urls = [url_for('display_image', filename=img) for img in crop_img_list]

            except Exception as e:
                flash(f"An error occurred: {str(e)}", "danger")
                return redirect(request.url)

    return render_template('home1.html', file_urls=file_urls)


@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(img_crop_folder, filename)


@app.route('/fetch_digit', methods=['GET', 'POST'])
def fetch_digit():
    global file_urls
    """Extract and recognize digits from the latest cropped image."""
    # Get the latest cropped image (assuming one image at a time)

    # Initial blank OTP digits
    # otp_digits = [''] * 8

    crop_img_list = sorted(os.listdir(img_crop_folder))
    if not crop_img_list:
        return render_template('home1.html', error="No cropped images found")

    latest_crop_path = os.path.join(img_crop_folder, crop_img_list[-1])

    # Step 1: Split digits
    digit_image_paths = split_image(latest_crop_path, split_digit_folder)

    if not digit_image_paths:
        return render_template('home1.html', error="No digits detected")

    # image_paths = os.listdir(split_digit_folder)

    image_paths = os.path.abspath(split_digit_folder)

    if not image_paths:
        return render_template('home1.html', error="No images found in the folder.")

    # Step 2: Recognize digits
    recognized_digits = digit_recognition(image_paths)

    # Step 3: Create 8 separate lists for each digit
    otp_digits = [str(digit) for digit in recognized_digits[:8]]  # Limit to 8 digits
    otp_digits += ['0'] * (8 - len(otp_digits))  # Fill the rest with '0' if less than 8 digits

    return render_template('home1.html', otp_digits=otp_digits, file_urls=file_urls)


@app.route('/back_to_homepage', methods=['POST', 'GET'])
def go_back_to_homepage():
    upload_path = os.path.abspath(display_folder)
    image_crop_path = os.path.abspath(img_crop_folder)
    split_image_path = os.path.abspath(split_digit_folder)
    try:
        for filename in os.listdir(upload_path):
            file_path = os.path.join(upload_path, filename)
            os.remove(file_path)
        for filename in os.listdir(image_crop_path):
            file_path = os.path.join(image_crop_path, filename)
            os.remove(file_path)
        for filename in os.listdir(split_image_path):
            file_path = os.path.join(split_image_path, filename)
            os.remove(file_path)
    except FileNotFoundError:
        return "File not found"

    return render_template('home1.html')


if __name__ == '__main__':
    app.run(debug=True)
