import cv2 
import numpy

image = cv2.imread('mqdefault.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result_image = image.copy()

letters = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 5 and h > 5:
        letter = image[y:y+h, x:x+w]
        letter = cv2.copyMakeBorder(letter, 2,2,2,2, cv2.BORDER_CONSTANT, value=[255,255,255])
        
        scale_factor = 20.0
        
        enlarged_image = cv2.resize(letter, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        # Отображение текущей буквы и запрос решения
        cv2.imshow('Current Letter', enlarged_image)
        key = cv2.waitKey(0)
        
        # Если нажата клавиша "s", то сохраняем букву
        if key == ord('s'):
            letters.append(letter)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else: continue
        
        cv2.destroyWindow('Current Letter')