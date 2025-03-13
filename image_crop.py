from imutils.perspective import four_point_transform
import numpy as np
import cv2


# Resize the image and set aspect ratio
def image_resizer(image, new_width=500):
    # Get height and width of the image
    h, w, c = image.shape
    new_height = int((h / w) * new_width)  # Calculate new height while maintaining aspect ratio
    size = (new_width, new_height)
    image_resized = cv2.resize(image, (new_width, new_height))  # Resize the image
    return image_resized, size


def img_crop(img_paths):
    # Load the original image
    img_original = cv2.imread(img_paths)

    if img_original is None:
        print("Error: Image not loaded. Check the file path.")
        return

    # Resize the image
    img_resized, size = image_resizer(img_original)

    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using Canny
    edge_image = cv2.Canny(blur, 75, 200)

    # Apply morphological dilation
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edge_image, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    four_points = None

    # Iterate through contours to find the one with 4 points (quadrilateral)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:  # Found a quadrilateral
            four_points = np.squeeze(approx)  # Get the 4 points as a flat array
            break

    if four_points is not None:
        # Draw the contour on the resized image
        cv2.drawContours(img_resized, [four_points], -1, (0, 255, 0), 3)

        # Find four points for the original image
        multiplier = img_original.shape[1] / size[0]
        four_points_original = four_points * multiplier
        four_points_original = four_points_original.astype(int)

        # Apply perspective transform to the original image
        wrap_image = four_point_transform(img_original, four_points_original)

        return wrap_image

    else:
        print("No quadrilateral found in the image.")
