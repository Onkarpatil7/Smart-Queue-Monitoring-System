import cv2

video_cap = cv2.VideoCapture(0)  # change to 1 or 2 if needed

if not video_cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    cv2.imshow("Webcam Feed", frame)
    # print(ret) //to check if frames are being captured


    if cv2.waitKey(10) == ord('q'): #0xff for 
        break

video_cap.release() #release camera for other usage
cv2.destroyAllWindows() #close webcam 
