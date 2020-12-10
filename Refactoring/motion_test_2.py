import cv2
import numpy
import module_image_2 as image
import matplotlib.pyplot as plt

count = 0
path_of_the_center = []
couple = [0] * 2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Параметры сглаживания m,n,
# параметр усреднения z, и массив среднего
m, n, z = 20, 20, 20
static_set = [0] * (z + 1)
# Проверяем, открывается ли видео
if not cap.isOpened():
    print("Error opening video stream or file")
x = []
y = []
# Массивы для кадров в движении
move_set = []
neural_network_set = []
# Открываем видео для чтения до окончания процесса
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        count += 1
        cv2.imshow('Frame', frame)
        if count > z + 1:
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            delta_duo = abs(Delta - old_delta)
            old_delta = Delta
            print(Delta)
            x.append(count)
            y.append(Delta)
            if Delta > 100 * medium_delta:
                move_set.append([frame, Delta])
                length_of_neural_set += 1
                movement_start = count
            else:
                if length_of_neural_set != 0:
                    if relax_time < 10:
                        relax_time += 1
                    else:
                        index_left = 0
                        index_right = length_of_neural_set
                        max_delta_left = 0
                        max_delta_right = 0
                        for k in range(length_of_neural_set):
                            if k < int(length_of_neural_set / 2):
                                if move_set[k][1] > max_delta_left:
                                    max_delta_left = move_set[k][1]
                                    index_left = k
                            else:
                                if move_set[k][1] > max_delta_right:
                                    max_delta_right = move_set[k][1]
                                    index_right = k
                        frame_left_max = move_set[index_left][0]
                        frame_medium = move_set[int(length_of_neural_set / 2)][0]
                        frame_right_max = move_set[index_right][0]
                        neural_network_set.append(
                            [frame_left_max, frame_medium, frame_right_max,
                             movement_start])
                        print('Тут должен быть мем')
                        length_of_neural_set = 0
                        relax_time = 0
                else:
                    if Delta < medium_delta * 40 and \
                            delta_duo < 1.5 * medium_duo:
                        static_set[z] = static_set[z] * z \
                                        - static_set[i][0]
                        static_set[i] = [couple[1], Delta]
                        i = (i + 1) % z
                        static_set[z] = (static_set[z] +
                                         static_set[i][0]) / z
                        couple[0] = static_set[z]
        elif count == z + 1:
            static_set[count - 2] = [couple[1], Delta]
            sum_delta = sum_delta + Delta
            medium_delta = sum_delta / z
            static_set[z] = static_set[z] * (count - 2)
            static_set[z] += static_set[count - 2][0]
            static_set[z] = static_set[z] / (count - 1)
            couple[0] = static_set[z]
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            old_delta = Delta
            medium_duo = medium_duo / z
            x.append(count)
            y.append(Delta)
        elif count == 1:
            height, width = frame.shape[:2]
            couple[0] = image.rgb(frame, m, n)
            x0 = int(width / 2)
            y0 = int(height / 2)
            x.append(count)
            # начальное значение используемых величин
            old_delta = 0
            medium_delta = 0
            sum_delta = 0
            medium_duo = 0
            motion = 0
            length_of_neural_set = 0
            relax_time = 0
            movement_start = 'None'
            i = 0
        else:
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            medium_duo += abs(Delta - old_delta)
            old_delta = Delta
            print(Delta)
            x.append(count)
            y.append(Delta)
            static_set[count - 2] = [couple[1], Delta]
            sum_delta = sum_delta + Delta
            static_set[z] = static_set[z] * (count - 2)
            static_set[z] += static_set[count - 2][0]
            static_set[z] = static_set[z] / (count - 1)
            couple[0] = static_set[z]
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Выход из видео, когда процесс окончен, построение графика
cap.release()
plt.plot(x[26:], y[25:], "bo")
plt.show()
cv2.destroyAllWindows()
