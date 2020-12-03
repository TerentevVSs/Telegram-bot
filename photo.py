# пример работы
import image
picrez = [0]*2
image.rgb("c2.png", picrez, 0)
image.rgb("c3.png", picrez, 1)
height, width = picrez[0].shape[:2]
M, x, y = image.determinant(picrez, height, width, 0, 1)
print(M,x,y)
