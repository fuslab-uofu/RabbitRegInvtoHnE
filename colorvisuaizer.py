import cv2
import numpy as np


# A function to handle mouse events
def click_event(event, x, y, flags, param):
    # Check if the event is a left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # x, y are the coordinates of the click.
        # Note: In OpenCV (numpy array), pixels are accessed as img[y, x] (row, column).
        pixel_values = img[y, x]
        print(f"Coordinates (x, y): ({x}, {y})")
        # OpenCV reads in BGR format by default.

        # Optional: Display the RGB values
        blue, green, red = pixel_values
        print(f"RGB Pixel values: ({red}, {green}, {blue})")

        # Optional: Put the values on the image itself
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"RGB:({red}, {green}, {blue})"
        cv2.putText(img, text, (x, y), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("image", img)


# Load an image
# Replace 'your_image.jpg' with the path to your image file
image_path = '/Users/jbonaventura/Downloads/R24-240/block04/CroppedImages/IMG_0053_scatter.tiff'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image from {image_path}. Check the file path.")
else:
    # Create a window and bind the function to the window
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()