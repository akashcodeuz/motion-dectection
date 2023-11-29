import cv2
import time
from datetime import datetime

# Create a VideoCapture object to capture video from a webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify a video file path

# Get the dimensions of the captured frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the size of the smaller square area
square_size = 200  # You can adjust the size as needed

# Calculate the coordinates for the smaller square area at the center
x = int(frame_width / 2 - square_size / 2)
y = int(frame_height / 2 - square_size / 2)
small_square_area = (x, y, square_size, square_size)

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Set the initial window size
cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Motion Detection', 800, 600)  # Adjust the size as needed

# Additional functionalities
motion_count = 0
start_time = time.time()
frame_count = 0
no_motion_timeout = 5  # Set the timeout duration in seconds
last_motion_time = time.time()
initial_no_motion = True  # Variable to track initial no motion
no_motion_display_timeout = 3  # Set the duration to display "No Motion Detected!" in seconds
no_motion_start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Extract the smaller square area from the frame
    x, y, w, h = small_square_area
    roi = frame[y:y+h, x:x+w]

    # Draw a green outline around the smaller square area
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Apply background subtraction to the smaller square area
    fgmask = fgbg.apply(roi)

    # Threshold the foreground mask to identify motion
    threshold = 50  # Adjust the threshold value as needed
    _, thresh = cv2.threshold(fgmask, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = any(cv2.contourArea(contour) > 100 for contour in contours)
    hand_movement = any(cv2.contourArea(contour) > 100 for contour in contours)

    # Check for initial no motion
    if initial_no_motion and motion_detected:
        initial_no_motion = False

    # Check for no motion
    current_time = time.time()
    if not initial_no_motion and not motion_detected and current_time - last_motion_time > no_motion_timeout:
        if current_time - no_motion_start_time < no_motion_display_timeout:
            cv2.putText(frame, "No Motion Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)  # Display "No Motion Detected!" text
            # Display a cross sign when no motion is detected
            cv2.line(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.line(frame, (x + w, y), (x, y + h), (0, 0, 255), 2)
        else:
            no_motion_start_time = time.time()

    # Update last motion time
    if motion_detected:
        last_motion_time = current_time

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
            # Get the coordinates of the detected contour and draw a green rectangle around it
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle
            break

    if motion_detected:
        cv2.putText(frame, "Aditya Lama Motion Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        motion_count += 1

    if hand_movement:
        cv2.putText(frame, "Hand Movement!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display total motions
    cv2.putText(frame, f"Total Motions: {motion_count}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)

    # Display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Display current time
    current_time_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {current_time_str}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)

    # Resize and place the smaller square area within the larger window
    frame[y:y+h, x:x+w] = cv2.resize(roi, (w, h))

    cv2.imshow('Motion Detection', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' key to exit
        break

    # Close the "No Motion Detected!" window after the specified timeout
    if current_time - no_motion_start_time >= no_motion_display_timeout:
        cv2.putText(frame, "No Motion Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)  # Display "No Motion Detected!" text

cap.release()
cv2.destroyAllWindows()
