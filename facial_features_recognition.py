# Import OpenCV Library
import cv2

# Loading the cascade files
front_face_cascade = cv2.CascadeClassifier('cascade_files/front_face_cascade.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/eye_cascade.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/nose_cascade.xml')


def face_detection(gray, frame):
    # faces are tuples, and contains x, y w, h as data for each tuple.
    # x & y are the coodinates for the upper left corner and w & h are width and height of the face frame.
    
    # Detect eyes
    faces = facial_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    # iterate through the faces to draw a rectangle around them
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        # Get the region of interest in the gray and frame, that could be face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect eyes
        eyes = eyes_cascade.detectMultiScale(image=roi_gray, scaleFactor=1.1, minNeighbors=22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img=roi_color, pt1=(ex, ey), pt2=(ex + ew, ey + eh), color=(0, 255, 0), thickness=2)
            
        # Detect nose
        nose = nose_cascade.detectMultiScale(image=roi_gray, scaleFactor=1.3, minNeighbors=10)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(img=roi_color, pt1=(nx, ny), pt2=(nx + nw, ny + nh), color=(0, 0, 255), thickness=2)
            
    return frame

# Turning the webcam on.
video_capture = cv2.VideoCapture(0)

# Repeat infinitely loop (until break):
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = face_detection(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Turning the webcam off.
video_capture.release()
# Destroy all the windows
cv2.destroyAllWindows()
