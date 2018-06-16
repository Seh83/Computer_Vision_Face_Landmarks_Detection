# Import OpenCV Library
import cv2

# Loading the cascade files
front_face_cascade = cv2.CascadeClassifier('cascade_files/front_face_cascade.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/eye_cascade.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/nose_cascade.xml')


def detect(gray, frame):
    faces = front_face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        region_of_interest_gray = gray[y:y+h, x:x+w]
        region_of_interest_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(region_of_interest_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(region_of_interest_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)

        nose = nose_cascade.detectMultiScale(region_of_interest_gray, 1.1, 7)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(region_of_interest_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
    return frame

# Turning the webcam on.
video_capture = cv2.VideoCapture(0)

# Repeat infinitely loop (until break):
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Turning the webcam off.
video_capture.release()
# Destroy all the windows
cv2.destroyAllWindows()
