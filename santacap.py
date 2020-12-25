import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# download the 'haarcascade_frontalface_default.xml' file from OpenCV Github repo and save in your PC
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while (cap.isOpened()):
    ret, frame = cap.read()
    height = 480
    cv2.rectangle(frame, (95, height - 80), (105, height - 5), (28, 66, 97),
                  15)
    # bigger triangle
    p1 = (15, height - 60)
    p2 = (185, height - 60)
    p3 = (100, height - 180)
    cv2.drawContours(frame, [(np.array([p1, p2, p3])).astype(int)], 0,
                     (46, 145, 0), -1)
    # middle triangle
    pm1 = (30, height - 120)
    pm2 = (170, height - 120)
    pm3 = (100, height - 210)
    cv2.drawContours(frame, [(np.array([pm1, pm2, pm3])).astype(int)], 0,
                     (46, 145, 0), -1)
    # top triangle
    pt1 = (45, height - 170)
    pt2 = (155, height - 170)
    pt3 = (100, height - 240)
    cv2.drawContours(frame, [(np.array([pt1, pt2, pt3])).astype(int)], 0,
                     (46, 145, 0), -1)
    # decorations
    cv2.circle(frame, (pt3), 5, (70, 235, 250), 20)
    cv2.circle(frame, (pt1), 5, (255, 255, 255), 10)
    cv2.circle(frame, (pm2), 5, (175, 47, 235), 10)
    cv2.circle(frame, (pm1), 5, (47, 128, 235), 10)
    cv2.circle(frame, (pt2), 5, (47, 97, 235), 10)
    cv2.circle(frame, (p2), 5, (255, 255, 255), 10)
    cv2.circle(frame, (p1), 5, (175, 47, 235), 10)

    cv2.circle(frame, (100, height - 100), 5, (70, 235, 250), 10)
    cv2.circle(frame, (100, height-160), 5, (70, 235, 250), 10)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # print(len(faces))
    if (len(faces) > 0):
        for (x, y, w, h) in faces:
            # drawing a santa cap
            cv2.drawContours(frame, [(np.array([(x, y), (x + (w / 2), y - 150),
                                                (x + w, y)])).astype(int)], 0,
                             (0, 0, 255), -1)
            cv2.circle(frame, (x + int((w / 2)), y - 160),
                       5, (255, 255, 255),
                       thickness=30)
            cv2.imshow('cap', frame)
    else:
        cv2.putText(frame, 'Detecting Face', (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        cv2.imshow('cap', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
