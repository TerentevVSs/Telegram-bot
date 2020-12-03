import cv2
import numpy as np

count = 0
path_of_the_center = []

def rgb_change_center(old_image, image):
  """
  Функция, сравнивая две фотографии, находит аналог центра масс для изменения RGB
  :param old_image:изображение с предыдущего кадра
  :param image:изображение на новом кадре
  :return:
  """
  (height, width, ch_num) = image.shape
  x_change = 0
  y_change = 0
  sum_change = 0
  for x in range(height):
    for y in range(width):
      old_blue = old_image.item(x, y, 0)
      old_green = old_image.item(x, y, 1)
      old_red = old_image.item(x, y, 2)
      blue = image.item(x, y, 0)
      green = image.item(x, y, 1)
      red = image.item(x, y, 0)
      rgb_change = ((green - old_green) ** 2 + (red - old_red) ** 2 + (green - old_green) ** 2) ** 0.5
      x_change += rgb_change * x
      y_change += rgb_change * y
      sum_change += rgb_change
  x_center = x_change / sum_change
  y_center = y_change / sum_change
  return (round(x_center), round(y_center))

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
      (x_center, y_center) = rgb_change_center(old_frame, frame)
      path_of_the_center.append((x_center, y_center))
      frame_to_show = cv2.circle(frame, (x_center, y_center), 1, 255, 1)
      cv2.imshow('Frame', frame_to_show)


    old_frame = frame

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