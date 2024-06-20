import cv2
import numpy as np

# Cargar la imagen original y la imagen de fondo
imagen_original = cv2.imread('imagen.jpg')
imagen_fondo = cv2.imread('imagen_fondo.jpeg')

# Asegurarse de que ambas imágenes tengan el mismo tamaño
imagen_fondo = cv2.resize(imagen_fondo, (imagen_original.shape[1], imagen_original.shape[0]))

# Convertir la imagen original a escala de grises
gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

# Aplicar umbral para obtener una máscara binaria de la imagen original
_, mascara = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY)

# Invertir la máscara para obtener el primer plano
mascara_inv = cv2.bitwise_not(mascara)

# Usar la máscara para extraer el primer plano de la imagen original
primer_plano = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara_inv)

# Usar la máscara invertida para extraer el fondo de la imagen de fondo
fondo = cv2.bitwise_and(imagen_fondo, imagen_fondo, mask=mascara)

# Combinar el primer plano y el fondo
resultado = cv2.add(primer_plano, fondo)

# Guardar la imagen resultante
cv2.imwrite('imagen_resultado.jpg', resultado)
