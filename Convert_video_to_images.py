# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
#cam = cv2.VideoCapture("/Users/zhangyuke/Desktop/ACADEMIC/SPRING_2022/Research-Computer_Vision/Film_Bag_Data/Single_Bag_film1.mp4")
cam = cv2.VideoCapture("/Users/zhangyuke/Desktop/ACADEMIC/SPRING_2022/Research-Computer_Vision/Film_Bag_Data/Single_Bag_film2.mp4")

# data1 stores images captured from Single_Bag_film1.mp4
# data2 stores images captured from Single_Bag_film2.mp4

try:

    # creating a folder named data
    if not os.path.exists('data2'):
        os.makedirs('data2')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

while (True):

    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = './data2/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()