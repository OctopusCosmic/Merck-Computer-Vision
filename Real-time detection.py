

import cv2
import time
def main():
    # Input
    source = 0
    # Define a window
    cv2.namedWindow("preview")
    # Start capture video by laptop
    camera = cv2.VideoCapture(source) # 0 for laptop camera, 1 for external camera
    # Get frame captured by camera
    ret, frame = camera.read()
    while ret:
        # Show frames
        cv2.imshow("preview", frame)
        # Get frame captured by camera
        ret, frame = camera.read()

        # Key to trigger stop capturing
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

        # use buffer to hold several frames
        # How many frames per seconds


    # Close the window
    cv2.destroyWindow("preview")
    # Close capturing device
    camera.release()

main()