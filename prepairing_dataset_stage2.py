import cv2
import numpy as np

color_flag = True

# Путь к папке с изображениями
image_folder = 'dataset/stage1/'

# Создание пустого массива для сохранения названий изображений
saved_image_names = []

# Начальный индекс изображения
image_index = 1000

# Коэффициент увеличения пикселей
scale_factor = 20

while True:
    # Формируем имя файла
    image_filename = f'element_{image_index}.png'
    image_path = image_folder + image_filename

    # Загрузка изображения
    image = cv2.imread(image_path)

    if image is not None:
        # Создание маски для черных пикселей
        black_mask = (image == [0, 0, 0]).all(axis=2)

        # Замена черных пикселей на белые
        image[black_mask] = [255, 255, 255]

        # Определение размера квадрата (берем максимальный размер между шириной и высотой)
        size = max(image.shape[0], image.shape[1])

        # Создание пустого квадратного изображения
        square_image = np.full((size, size, 3), 255, dtype=np.uint8)

        # Определение координат для вставки исходного изображения по центру квадрата
        x_offset = (size - image.shape[1]) // 2
        y_offset = (size - image.shape[0]) // 2

        # Вставка исходного изображения в центр квадрата
        square_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

        if color_flag:
            # Конвертация в оттенки серого
            square_image = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)

        # Уменьшение яркости всех пикселей, кроме белых
        mask = square_image < 230
        square_image[mask] = square_image[mask] * 0.3  # Например, уменьшим яркость вдвое


        # Увеличение изображения
        enlarged_image = cv2.resize(square_image, (28, 28))
        
        # Увеличение размера пикселей
        enlarged_image1 = cv2.resize(enlarged_image, (28 * scale_factor, 28 * scale_factor), interpolation=cv2.INTER_NEAREST)

        # Отображение увеличенного изображения
        cv2.imshow('Enlarged Image', enlarged_image1)

        # Ожидание нажатия клавиши
        key = cv2.waitKey(0)

        # Если нажата клавиша 'Y', переходим к следующему изображению
        if key == ord('y'):
            image_index += 1

        # Если нажата клавиша 'U', сохраняем название изображения в массиве
        elif key == ord('u'):
            cv2.imwrite(f'dataset/stage2/processed_{image_index}.png', enlarged_image)
            image_index += 1

        # Если нажата клавиша 'Esc', выходим из цикла
        elif key == ord('q'):
            break

    else:
        # Если изображение с заданным индексом не найдено, выходим из цикла
        break

# Выводим названия сохраненных изображений
print("Сохраненные изображения:")
for name in saved_image_names:
    print(name)

# Закрываем окно OpenCV
cv2.destroyAllWindows()
