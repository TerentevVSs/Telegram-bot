import cv2
import numpy as np
import image
count = 0
path_of_the_center = []
couple = [0]*2
cap = cv2.VideoCapture('chaplin.mp4')

# Проверяем, открывается ли видео
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Открываем видео для чтения до окончания процесса
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    count += 1
    if count >= 2:
      couple[0], couple[1] = couple[1], couple[0]
      image.rgb(frame, couple, 1, height, width)
      Delta = image.delta(couple)
      print(Delta)
    else:
      height, width = frame.shape[:2]
      image.rgb(frame, couple, 1, height, width)
      x0 = int(width/2)
      y0 = int(height/2)
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

cv2.destroyAllWindows()
print(path_of_the_center)