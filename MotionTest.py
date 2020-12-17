import cv2
import module_image as image
import matplotlib.pyplot as plt
import Neural_Network
import numpy as np
count = 0
path_of_the_center = []
couple = [0] * 2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Параметры сжатия m,n
m = 20
n = 20
# параметр кол-ва кадров в усреднении len_static_set, и массив кадров для усреднения  static_set  + средний кадр
len_static = 40
static_set = [0] * (len_static + 1)
# позиция в static_set на который будет поставлен новый статичный кадр
position_static_set = 0
# Проверяем, открывается ли видео
if (cap.isOpened() == False):
    print("Error opening video stream or file")
x = []
y = []
# neural_network_set кадр для проверки
neural_network_check = "None"
# длина списка набор для нейросети
len_neur_set = 0
# medium_delta - среднее отклонение в покое
medium_delta = 0
# величина для подсчёта medium_delta
sum_delta = 0
# среднее отклонение двух последовательных кадров друг от друга
medium_duo = 0
# кол-во кадров после конца движения
relax_time = 0
# статус движение - покой
status = "static"
worry = 0
# Открываем видео для чтения до окончания процесса
mode = "day"
while (cap.isOpened()):
    ret, frame = cap.read()
    if mode == "day":
        # параметры для дня
        param1 = 70
        param2 = 5
        param3 = 50
        param4 = 2
    else:
        # параметры ночи
        param1 = 40
        param2 = 0
        param3 = 15
        param4 = 2
    if ret == True:
        count += 1
        if count > len_static + 1:
            old_frame = couple[1]
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            delta_duo = image.delta([couple[1], old_frame])
            print(Delta)
            x.append(count)
            y.append(Delta)
            if Delta > medium_delta*param1  and delta_duo > medium_duo * param2:
                status = "move"
                next = next + 1
                if next%3 == 0:
                    neural_network_check = frame
                    neural_network_check = np.array(neural_network_check, dtype='uint8')
                    neural_network_check = cv2.resize(neural_network_check, dsize=(160, 90), interpolation=cv2.INTER_CUBIC)
                    neural_network_check = neural_network_check.reshape(160*90*3, 1)/255
                    worry = Neural_Network.check_image(neural_network_check)
                    if worry == 1:
                        # Тут вывод в бота
                        print("Человек", count)
                        worry = 0
            else:
                # проверка было ли движение
                if status == "move":
                    # движение было, проверяет завершённость
                    relax_time += 1
                    if relax_time == 10:
                        status = "static"
                else:
                    # движение не было иди прошло, проверка для добавление кадр в усреднение
                    if Delta < param3 * medium_delta and delta_duo < medium_duo * param4:
                        # кадр подходит для усреднения
                        static_set[len_static] = static_set[len_static] * len_static \
                                       - static_set[position_static_set][0]
                        static_set[position_static_set] = [couple[1], Delta]
                        static_set[len_static] = (static_set[len_static] +
                                        static_set[position_static_set][0]) / len_static
                        couple[0] = static_set[len_static]
                        position_static_set = (position_static_set + 1) % len_static
        elif count == len_static + 1:
            static_set[count - 2] = [couple[1], Delta]
            # вычислениек medium_delta
            sum_delta = sum_delta + Delta
            medium_delta = sum_delta / len_static
            # обновление среднего кадра
            static_set[len_static] = static_set[len_static] * (count - 2)
            static_set[len_static] += static_set[count - 2][0]
            static_set[len_static] = static_set[len_static] / (count - 1)
            couple[0] = static_set[len_static]
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            # вычисление medium_duo
            medium_duo += image.delta([static_set[count - 2][0], static_set[count - 3][0]])
            medium_duo = medium_duo / len_static
            x.append(count)
            y.append(Delta)
        elif count == 1:
            height, width = frame.shape[:2]
            couple[0] = image.rgb(frame, m, n)
            x0 = int(width / 2)
            y0 = int(height / 2)
            x.append(count)
            next = 0

        else:
            couple[1] = image.rgb(frame, m, n)
            Delta = image.delta(couple)
            print(Delta)
            x.append(count)
            y.append(Delta)
            static_set[count - 2] = [couple[1], Delta]
            # подсчёт
            if count > 2:
                medium_duo += image.delta([static_set[count - 2][0], static_set[count-3][0]])
            sum_delta = sum_delta + Delta
            # обновление среднего кадра
            static_set[len_static] = static_set[len_static] * (count - 2)
            static_set[len_static] += static_set[count - 2][0]
            static_set[len_static] = static_set[len_static] / (count - 1)
            couple[0] = static_set[len_static]
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
