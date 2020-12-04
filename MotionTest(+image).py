import cv2
import numpy as np
import module_image as image
import matplotlib.pyplot as plt

count = 0
path_of_the_center = []
couple = [0] * 2
cap = cv2.VideoCapture('test2.mp4')

# Проверяем, открывается ли видео
if (cap.isOpened() == False):
    print("Error opening video stream or file")
x = []
y = []
sum = 0
# Открываем видео для чтения до окончания процесса
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        count += 1
        if count > 21:
            couple[1] = image.rgb(frame, height, width)
            Delta = image.delta(couple)
            print(Delta)
            x.append(count)
            y.append(Delta)
        elif count == 21:
            average = sum / 20
            couple[0] = average
            couple[1] = image.rgb(frame, height, width)
            Delta = image.delta(couple)
            x.append(count)
            y.append(Delta)
        elif count == 1:
            height, width = frame.shape[:2]
            couple[0] = image.rgb(frame, height, width)
            x0 = int(width / 2)
            y0 = int(height / 2)
            Delta = image.delta(couple)
            x.append(count)
            sum += couple[1]
        else:
            height, width = frame.shape[:2]
            couple[1] = image.rgb(frame, height, width)
            x0 = int(width / 2)
            y0 = int(height / 2)
            Delta = image.delta(couple)
            x.append(count)
            y.append(Delta)
            sum += couple[1]
        frame_to_show = cv2.circle(frame, (x0, y0), 1, 255, 1)
        cv2.imshow('Frame', frame_to_show)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Выход из видео, когда процесс окончен
cap.release()
plt.plot(x[26:], y[25:], "bo")
plt.show()
cv2.destroyAllWindows()
