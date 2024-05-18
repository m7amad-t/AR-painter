import cv2
import mediapipe as mp
import numpy as np



width = 640
height = 480




# Initialize MediaPipe for hand tracking
hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1
)


drawing_utils = mp.solutions.drawing_utils  # For drawing landmarks

# Constants for drawing
DRAW_THICKNESS = 5  # Thickness of the lines to draw

# Define a dictionary to map color keys to BGR values
color_options = {
    'r': (0, 0, 255),  # Red
    'b': (255, 0, 0),  # Blue
    'g': (0, 255, 0),  # Green
    'k': (0, 255, 255)  # Yellow
}

# Default color for drawing (red)
current_color = color_options['r']

# Create a mask to draw on
# Create a canvas for drawing with full-screen size
mask = np.ones((height, width, 3), dtype=np.uint8) * 0

previous_x, previous_y = None, None  # Previous drawing position

# Webcam setup
cap = cv2.VideoCapture(0)

cap.set(3 , width)
cap.set(4 , height)


def count_raised_fingers(landmarks):
    # Indices for fingertips and corresponding knuckles
    fingertip_indices = [4, 8, 12, 16, 20]
    knuckle_indices = [3, 6, 10, 14, 18]

    # Count raised fingers by checking if fingertips are above knuckles
    count = 0
    for tip, knuckle in zip(fingertip_indices, knuckle_indices):
        if landmarks.landmark[tip].y <= landmarks.landmark[knuckle].y:
            count += 1
    return count




def drawRecs (frame):
    cv2.rectangle(frame, (200, 0), (260, 40), color_options['r'], -1)  # Display the current color
    cv2.rectangle(frame, (265, 0), (330, 40), color_options['g'], -1)  # Display the current color
    cv2.rectangle(frame, (335, 0), (400, 40), color_options['b'], -1)  # Display the current color
    cv2.rectangle(frame, (405, 0), (470, 40), color_options['k'], -1)  # Display the current color


def detectColor (x_pos , y_pos ) :
    print(x_pos)
    if y_pos > 40:
        return None
    if x_pos >= 200 and x_pos <= 260 :
        return color_options['r']
    elif x_pos >= 265 and x_pos <= 330:
        return color_options['g']
    elif x_pos >= 335 and x_pos <= 400:
        return color_options['b']
    elif x_pos >= 405 and x_pos <= 470:
        return color_options['k']


# Main loop for drawing application
while True:
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        break



    frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(rgb_frame)



    if results.multi_hand_landmarks:


        # If there are hand landmarks detected
        for hand_landmarks in results.multi_hand_landmarks:


            # Draw landmarks for visualization
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            # Get the index fingertip position
            index_x = int(hand_landmarks.landmark[8].x * width)
            index_y = int(hand_landmarks.landmark[8].y * height)

            # Count the number of raised fingers
            num_fingers = count_raised_fingers(hand_landmarks)
            if num_fingers == 2:
                # means the index finger is raised so draw
                if previous_x is not None and previous_y is not None:
                    # Draw a line on the mask with the current color
                    cv2.line(mask, (previous_x, previous_y), (index_x, index_y), current_color, DRAW_THICKNESS)

                # Update previous coordinates
                previous_x, previous_y = index_x, index_y
            elif num_fingers > 3 :
                # Erase by drawing white circles on the canvas
                cv2.circle(mask, (index_x, index_y), 30, (0, 0, 0), -1)  # Erasing on canvas white
                cv2.circle(frame, (index_x, index_y), 30, (255, 255, 255), -1)  # Erasing with white
                previous_x, previous_y = None, None  # Reset previous point
            elif num_fingers == 3 :
                color  = detectColor(index_x, index_y)
                if color is not None:
                    current_color = color
                previous_x, previous_y = None, None
            else :
                previous_x, previous_y = None , None


    merged_frame = cv2.add(frame , mask)

    # Display instructions for changing colors
    cv2.putText(merged_frame, "Current Color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Show the current drawing color
    cv2.rectangle(merged_frame, (110,15), (160, 40), current_color, -1)  # Display the current color

    # drawing the color rectangles
    drawRecs(merged_frame)

    # Display the final frame
    cv2.imshow("Drawing App", merged_frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # Exit on 'Esc' or 'q'
        break


# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
