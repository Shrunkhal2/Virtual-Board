import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm  # Custom hand tracking module

# Constants
brushThickness = 25
eraserThickness = 100

# Load header images for the virtual board
folderPath = "Header"  # Folder containing header images
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

# Set default header and draw color
header = overlayList[0]
drawColor = (255, 0, 255)  # Default color (pink)

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0  # Previous points
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # Canvas for drawing

while True:
    # 1. Capture image from webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip image horizontally

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Get tip positions of index and middle fingers
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. Selection Mode: Two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0  # Reset drawing coordinates
            print("Selection Mode")
            if y1 < 125:  # Check if in header region
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)  # Pink
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)  # Blue
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)  # Green
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  # Eraser
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. Drawing Mode: Only index finger up
        if fingers[1] and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw on canvas
            if drawColor == (0, 0, 0):  # Eraser
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Merge canvas and live video feed
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Set header image
    img[0:125, 0:1280] = header

    # Display the result
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()