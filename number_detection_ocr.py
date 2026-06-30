import cv2
import easyocr
import numpy as np

# Initialize the OCR reader for English
# gpu=False means it uses CPU (set True if you have a compatible GPU)
reader = easyocr.Reader(['en'], gpu=False)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():

    # Read a frame from webcam
    success, img = cap.read()

    # Exit if frame reading fails
    if not success:
        break

    # Run OCR on the current frame
    # detail=1 gives bounding box + text + confidence score
    results = reader.readtext(img)

    # Loop through each detected text region
    for (bbox, text, confidence) in results:

        # Filter only numbers using isdigit()
        # text.strip() removes spaces, isdigit() checks if all chars are digits
        if text.strip().isdigit():

            # bbox contains 4 corner points of the detected text box
            # Convert to integer coordinates for drawing
            top_left     = tuple(map(int, bbox[0]))  # top-left corner
            bottom_right = tuple(map(int, bbox[2]))  # bottom-right corner

            # Draw a green rectangle around the detected number
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

            # Show the detected number and confidence above the rectangle
            label = f'{text}  ({confidence*100:.0f}%)'
            cv2.putText(img, label, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Print to console as well
            print(f'Number detected: {text}  Confidence: {confidence*100:.1f}%')

    # Show the frame
    cv2.imshow('Number Detection OCR', img)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()