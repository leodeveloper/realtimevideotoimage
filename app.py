import cv2

def capture_image():
    # Open a connection to the default camera (usually the first camera detected)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Set camera resolution if needed (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
    else:
        # Display the captured frame (optional)
        cv2.imshow('Captured Image', frame)
        
        # Wait for a key press (optional)
        cv2.waitKey(0)
        
        # Save the captured frame to a file
        cv2.imwrite('captured_image.jpg', frame)
        print("Image saved as 'captured_image.jpg'.")

    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
