import telebot
from telebot import types

bot = telebot. TeleBot("1427672157:AAH_K0LbWsIuNyUR3vQC9tEvxxrkHlqVEuY")

keyboard1 = telebot.types.ReplyKeyboardMarkup(True, True)
keyboard1.row('Привет', 'Пока')

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, начнем видеонаблюдение?', reply_markup=keyboard1)



@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text == 'Привет':
        bot.send_message(message.chat.id, 'Привет')
        bot.send_sticker(message.chat.id, 'CAADAgADZgkAAnlc4gmfCor5YbYYRAI')
    elif message.text == 'Пока':
        bot.send_message(message.chat.id, 'Прощай')

@bot.message_handler(content_types = ['sticker'])
def sticker_id(message):
    print(message)

#@client.message_handler(commands = ['get_info', 'info'])
#def get_user_info(message):
   # if message.text.lower() == '/pep':
     #   keyboard = types.InlineKeyboardMarkup();  # наша клавиатура
   #     key_yes = types.InlineKeyboardButton(text='Да', callback_data='yes');  # кнопка «Да»
   #     keyboard.add(key_yes);  # добавляем кнопку в клавиатуру
  #      key_no = types.InlineKeyboardButton(text='Нет', callback_data='no');
  #      keyboard.add(key_no);
  #      question = 'У тебя есть деньги?';
  #      bot.send_message(message.from_user.id, text=question, reply_markup=keyboard)


#@bot.callback_query_handler(func = lambda call: True)
#def answer(call):
    #if call.data == 'yes':
   #     markup_reply = types.ReplyKeyboardMarkup(resize_keyboard = True)
   #     item_id = types.KeyboardButton('МОЙ ID')
   #     item_username = types.KeyboardButton('МОЙ НИК')

    #    markup_reply.add(item_id, item_username)
    #    bot.send_message(call.message.chat.id, 'Нажмите на одну из кнопок',
    #                     reply_markup = markup_reply
    #    )
   # elif call.data == 'no':
   #     pass


#@bot.message_handler(content_types = ['text'])
#def get_text(message):
 #   if message.text == 'МОЙ ID':
 #       bot.send_message(message.chat.id, f'Your ID: {message.from_user.id}')
  #  elif message.text == 'МОЙ НИК':
  #      bot.send_message(message.chat.id, f'Your ID: {message.from_user.first_name} {message.from_user.second_name}')

#def start(message):
  # if message.text.lower() == '/start':
   #     bot.send_message(message.chat.id, "Как тебя зовут?")
   #     bot.register_next_step_handler(message, get_name) #следующий шаг – функция get_name

#def get_name(message): #получаем фамилию
  #  global name
  #  name = message.text
  #  bot.send_message(message.chat.id, 'Какая у тебя фамилия?')
  #  bot.register_next_step_handler(message, get_surname)

#def get_surname(message):
  #  global surname
  #  surname = message.text
  #  bot.send_message(message.from_user.id, 'Сколько тебе лет?')
  #  bot.register_next_step_handler(message, get_age)

#def get_age(message):
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

#@bot.callback_query_handler(func=lambda call: True)
#def callback_worker(call):
   # if call.data == "yes": #call.data это callback_data, которую мы указали при объявлении кнопки
     #   ...                       #код сохранения данных, или их обработки
     #   bot.send_message(call.message.chat.id, 'Запомню : )')
  #  elif call.data == "no":
      #   ... #переспрашиваем

bot.polling(none_stop = True, interval = 0)