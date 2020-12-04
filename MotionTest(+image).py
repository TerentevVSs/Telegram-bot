import cv2
import numpy as np
import module_image as image
import matplotlib.pyplot as plt
count = 0
path_of_the_center = []
couple = [0] * 3
cap = cv2.VideoCapture('test.mp4')

# Проверяем, открывается ли видео
if (cap.isOpened() == False):
    print("Error opening video stream or file")
x=[]
y=[]
# Открываем видео для чтения до окончания процесса
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        count += 1
        if count >= 2:
            image.rgb(frame, couple, 1, height, width)
            Delta = image.delta(couple)
            print(Delta)
            x.append(count)
            y.append(Delta)
        else:
            height, width = frame.shape[:2]
            image.rgb(frame, couple, count, height, width)
            x0 = int(width / 2)
            y0 = int(height / 2)
            x.append(count)
        frame_to_show = cv2.circle(frame, (x0, y0), 1, 255, 1)
        cv2.imshow('Frame', frame_to_show)
        print(count)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Выход из видео, когда процесс окончен
cap.release()
plt.plot(x[1:], y, "bo")
plt.show()
cv2.destroyAllWindows()
