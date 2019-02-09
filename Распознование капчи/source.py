import cv2
from urllib import request
import numpy as np
import os
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image

def import_img (url, name_img):     #функция загрузки капчи
    f = open(name_img, 'wb')
    f.write(request.urlopen(url).read())
    f.close()
    return

def captcha_parsing(url):           #функция обработки и разбиения капчи на 5 изображений(по одной цифре)
    import_img(url, "captcha.png")
    block_size = 71
    offset = 10

    #Убирается шум из капчи
    Igray = cv2.imread("captcha.png")
    Igray = cv2.cvtColor(Igray, cv2.COLOR_RGB2GRAY)
    Igray = cv2.adaptiveThreshold(Igray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, offset)
    kernel = np.ones((3, 3), np.uint8)
    Igray = cv2.dilate(Igray, kernel=kernel)

    #Деление обработанной капчи  на цифры
    x = 28
    y = 5
    width = 19
    height = 30
    for i in range(5):
        img = Igray[y:y+height, x:x+width]
        img = cv2.resize(img, (28, 28))
        string = str(i) + ".jpg"
        cv2.imwrite(string, img)
        x += 19
    os.remove("captcha.png")        #Она уже не нужна, т.к. уже разбили и сохранили все цифры
    return

print("Введите ссылку изображения")     #Ввод ссылки не важен, главное сама ссылка
url = input()
captcha_parsing(url)
#Сейчас храниться 5 картинок с цифрами

# Загружаем данные об архитектуре сети из файла json
json_file = open("my_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
model = model_from_json(loaded_model_json)
# Загружаем веса в модель
model.load_weights("my_model.h5")
# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

arr = ""                            #Строчка, которая будет являтся ответом
for i in range(5):
    img = image.load_img(str(i) + ".jpg")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    arr += str(np.argmax(prediction) + 1)           #добавлляю элеиент в массив
    os.remove(str(i) + ".jpg")                      #После обработки цифры, ее изображение не нужно

print(arr)
