import telebot
from telebot import types
import cv2
import module_image as image
import Neural_Network
import numpy as np

FPS = 30
# Булева функция дня и ночи
it_is_day = None
# Словарь параметров дня и ночи
day = {"delta_high": 70, "duo_high": 5, "delta_low": 50, "duo_low": 2}
night = {"delta_high": 40, "duo_high": 0, "delta_low": 15, "duo_low": 2}

bot = telebot.TeleBot("1427672157:AAH_K0LbWsIuNyUR3vQC9tEvxxrkHlqVEuY")

keyboard1 = telebot.types.ReplyKeyboardMarkup(True, True)
keyboard1.row('Начнем', 'Не нужно')
keyboard2 = telebot.types.ReplyKeyboardMarkup(True, True)
keyboard2.row('День', 'Ночь')
keyboard3 = telebot.types.ReplyKeyboardMarkup(True, True)
keyboard3.row()


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, начнем видеонаблюдение?', reply_markup=keyboard1)


@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text == 'Начнем':
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAKGY1_bde_LilNGDnjcXIBFL93uqlwEAAJiFgAC6VUFGKq9bs1J0e3SHgQ')
        bot.send_message(message.chat.id, 'Выберите режим', reply_markup=keyboard2)
        bot.register_next_step_handler(message, choose_mode)  # следующий шаг – функция  choose_mode
    elif message.text == 'Не нужно':
        bot.send_message(message.chat.id, 'Прощай')


def choose_mode(message):
    global it_is_day
    if message.text == 'День':
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAKGY1_bde_LilNGDnjcXIBFL93uqlwEAAJiFgAC6VUFGKq9bs1J0e3SHgQ')
        it_is_day = True
        bot.register_next_step_handler(message, lets_work)  # следующий шаг – функция  lets_work
    elif message.text == 'Ночь':
        it_is_day = False
        bot.register_next_step_handler(message, lets_work)


def lets_work(message):
    count = 0
    checks_per_second = 10
    danger_time = 0
    bot_relax_time = FPS
    couple = [0] * 2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Параметры сжатия m,n
    m = 20
    n = 20
    # параметр количества кадров в усреднении len_static_set
    # и массив кадров для усреднения  static_set  + средний кадр
    len_static = 40
    static_set = [0] * (len_static + 1)
    # позиция в static_set на который будет поставлен новый статичный кадр
    position_static_set = 0
    # Проверяем, открывается ли видео
    if not cap.isOpened():
        print("Error opening video stream or file")
    # neural_network_set кадр для проверки
    neural_network_check = "None"
    # длина списка набор для нейросети
    len_neural_set = 0
    # medium_delta - среднее отклонение в покое
    medium_delta = 0
    # величина для подсчёта medium_delta
    sum_delta = 0
    # среднее отклонение двух последовательных кадров друг от друга
    medium_duo = 0
    # кол-во кадров после конца движения
    relax_time = 0
    # статус движение - покой
    it_moves = False
    # Статус движения после обработки нейросетью
    worry = 0
    # Открываем видео для чтения до окончания процесса
    while cap.isOpened():
        ret, frame = cap.read()
        if it_is_day:
            params = day
        else:
            params = night
        if ret:
            count += 1
            if count > len_static + 1:
                old_frame = couple[1]
                couple[1] = image.rgb(frame, m, n)
                Delta = image.delta(couple)
                delta_duo = image.delta([couple[1], old_frame])
                if Delta > medium_delta * (params["delta_high"] +
                                           medium_duo / delta_duo) \
                        and delta_duo > medium_duo * params["duo_high"]:
                    it_moves = True
                    if count % (FPS // checks_per_second) == 0:
                        neural_network_check = frame
                        neural_network_check = np.array(
                            neural_network_check, dtype='uint8')
                        neural_network_check = cv2.resize(
                            neural_network_check, dsize=(160, 90),
                            interpolation=cv2.INTER_CUBIC)
                        neural_network_check = neural_network_check.reshape(
                            160 * 90 * 3, 1) / 255
                        worry = Neural_Network.check_image(neural_network_check)
                        if worry == 1:
                            # Первая тревога
                            if danger_time == 0:
                                bot.send_message(message.chat.id, 'Тревога! Обнаружен человек!')
                                bot.send_photo(message.chat.id, frame)
                                danger_time = count
                            # Последующие тревоги
#                            elif danger_time > 0 and (count-danger_time) % bot_relax_time == 0:
#                                bot.send_photo(message.chat.id, frame)
                        if danger_time > 0:
                            # Окончание тревоги
                            bot.send_photo(message.chat.id, frame)
                            bot.send_message(message.chat.id,
                                             'Движение человека в кадре прекратилось')
                            danger_time = 0
                else:
                    # проверка было ли движение
                    if it_moves:
                        # движение было, проверяет завершённость
                        relax_time += 1
                        if relax_time == 10:
                            it_moves = False
                    else:
                        # движение не было иди прошло, проверка для
                        # добавление кадр в усреднение
                        if Delta < params["delta_low"] * medium_delta \
                                and delta_duo < medium_duo * params["duo_low"]:
                            # кадр подходит для усреднения
                            static_set[len_static] = static_set[len_static] \
                                                     * len_static \
                                                     - static_set[position_static_set][0]
                            static_set[position_static_set] = [couple[1],
                                                               Delta]
                            static_set[len_static] = (static_set[len_static] +
                                                      static_set[position_static_set][0]) / \
                                                     len_static
                            couple[0] = static_set[len_static]
                            position_static_set = (position_static_set + 1) \
                                                  % len_static
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
                medium_duo += image.delta([static_set[count - 2][0],
                                           static_set[count - 3][0]])
                medium_duo = medium_duo / len_static
            elif count == 1:
                height, width = frame.shape[:2]
                couple[0] = image.rgb(frame, m, n)
            else:
                couple[1] = image.rgb(frame, m, n)
                Delta = image.delta(couple)
                static_set[count - 2] = [couple[1], Delta]
                # подсчёт
                if count > 2:
                    medium_duo += image.delta([static_set[count - 2][0],
                                               static_set[count - 3][0]])
                sum_delta = sum_delta + Delta
                # обновление среднего кадра
                static_set[len_static] = static_set[len_static] * (count - 2)
                static_set[len_static] += static_set[count - 2][0]
                static_set[len_static] = static_set[len_static] / (count - 1)
                couple[0] = static_set[len_static]
            cv2.imshow('Frame', frame)
            # Press "space" on keyboard to  exit
            if cv2.waitKey(32) & 0xFF == ord(' '):
                break
        else:
            break
    # Выход из видео, когда процесс окончен
    cap.release()
    cv2.destroyAllWindows()


@bot.message_handler(content_types=['sticker'])
def sticker_id(message):
    print(message)


# @client.message_handler(commands = ['get_info', 'info'])
# def get_user_info(message):
# if message.text.lower() == '/pep':
#   keyboard = types.InlineKeyboardMarkup();  # наша клавиатура
#     key_yes = types.InlineKeyboardButton(text='Да', callback_data='yes');  # кнопка «Да»
#     keyboard.add(key_yes);  # добавляем кнопку в клавиатуру
#      key_no = types.InlineKeyboardButton(text='Нет', callback_data='no');
#      keyboard.add(key_no);
#      question = 'У тебя есть деньги?';
#      bot.send_message(message.from_user.id, text=question, reply_markup=keyboard)


# @bot.callback_query_handler(func = lambda call: True)
# def answer(call):
# if call.data == 'yes':
#     markup_reply = types.ReplyKeyboardMarkup(resize_keyboard = True)
#     item_id = types.KeyboardButton('МОЙ ID')
#     item_username = types.KeyboardButton('МОЙ НИК')

#    markup_reply.add(item_id, item_username)
#    bot.send_message(call.message.chat.id, 'Нажмите на одну из кнопок',
#                     reply_markup = markup_reply
#    )
# elif call.data == 'no':
#     pass


# @bot.message_handler(content_types = ['text'])
# def get_text(message):
#   if message.text == 'МОЙ ID':
#       bot.send_message(message.chat.id, f'Your ID: {message.from_user.id}')
#  elif message.text == 'МОЙ НИК':
#      bot.send_message(message.chat.id, f'Your ID: {message.from_user.first_name} {message.from_user.second_name}')

# def start(message):
# if message.text.lower() == '/start':
#     bot.send_message(message.chat.id, "Как тебя зовут?")
#     bot.register_next_step_handler(message, get_name) #следующий шаг – функция get_name

# def get_name(message): #получаем фамилию
#  global name
#  name = message.text
#  bot.send_message(message.chat.id, 'Какая у тебя фамилия?')
#  bot.register_next_step_handler(message, get_surname)

# def get_surname(message):
#  global surname
#  surname = message.text
#  bot.send_message(message.from_user.id, 'Сколько тебе лет?')
#  bot.register_next_step_handler(message, get_age)

# def get_age(message):
#  global age;
#  while age == 0: #проверяем что возраст изменился
#     try:
#            age = int(message.text) #проверяем, что возраст введен корректно
#      except Exception:
#           bot.send_message(message.from_user.id, 'Цифрами, пожалуйста');
#  keyboard = types.InlineKeyboardMarkup(); #наша клавиатура
# key_yes = types.InlineKeyboardButton(text='Да', callback_data='yes'); #кнопка «Да»
#  keyboard.add(key_yes); #добавляем кнопку в клавиатуру
# key_no= types.InlineKeyboardButton(text='Нет', callback_data='no');
#  keyboard.add(key_no);
#  question = 'У тебя есть деньги?';
#  bot.send_message(message.from_user.id, text=question, reply_markup=keyboard)

# @bot.callback_query_handler(func=lambda call: True)
# def callback_worker(call):
# if call.data == "yes": #call.data это callback_data, которую мы указали при объявлении кнопки
#   ...                       #код сохранения данных, или их обработки
#   bot.send_message(call.message.chat.id, 'Запомню : )')
#  elif call.data == "no":
#   ... #переспрашиваем

if __name__ == '__main__':
    bot.polling(none_stop=True)