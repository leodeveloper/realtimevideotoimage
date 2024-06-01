import cv2
import numpy as np
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def detect_camera_movement():
    # Open a connection to the default camera (usually the first camera detected)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and convert it to grayscale
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create some random points to track
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7))

    while True:
        # Capture a new frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is None or st is None:
            print("Warning: Optical flow calculation failed.")
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7))
            continue

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Calculate the mean displacement
        displacement = good_new - good_old
        mean_displacement = np.mean(displacement, axis=0)

        # Determine direction
        directions = []
        if mean_displacement[0] > 2:
            directions.append("right")
        elif mean_displacement[0] < -2:
            directions.append("left")

        if mean_displacement[1] > 2:
            directions.append("up")
        elif mean_displacement[1] < -2:
            directions.append("down")

        # Calculate mean point size change for forward and backward movement
        mean_sizes = []
        for i in range(len(good_new)):
            old_size = cv2.norm(good_old[i])
            new_size = cv2.norm(good_new[i])
            size_change = new_size - old_size
            mean_sizes.append(size_change)
        mean_size_change = np.mean(mean_sizes)

        if mean_size_change > 2:
            directions.append("backward")
        elif mean_size_change < -2:
            directions.append("forward")

        if not directions:
            directions.append("steady")

        # Display the direction on the frame
        direction_text = ", ".join(directions)
        cv2.putText(frame, f"Camera is moving {direction_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Speak the direction
        #speak(f"Camera is moving {direction_text}")


        # Display the frame
        cv2.imshow('Video', frame)

        # Check if 'q' key is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_camera_movement()
