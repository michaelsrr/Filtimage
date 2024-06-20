import cv2
import numpy as np

# Cargar la imagen original y la imagen de fondo
imagen_original = cv2.imread('imagen.jpg')
imagen_fondo = cv2.imread('imagen_fondo.jpeg')

# Asegurarse de que ambas imágenes tengan el mismo tamaño
imagen_fondo = cv2.resize(imagen_fondo, (imagen_original.shape[1], imagen_original.shape[0]))

# Cargar el clasificador en cascada para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detectar rostros en la imagen original
rostros = face_cascade.detectMultiScale(imagen_original, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Crear una máscara de la misma forma que la imagen original
mascara = np.zeros_like(imagen_original)

# Para cada rostro detectado, dibujar un rectángulo blanco en la máscara
for (x, y, w, h) in rostros:
    mascara[y:y+h, x:x+w] = [255, 255, 255]

# Convertir la máscara a escala de grises
mascara_gris = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)

# Aplicar umbral para obtener una máscara binaria
_, mascara_binaria = cv2.threshold(mascara_gris, 120, 255, cv2.THRESH_BINARY)

# Usar la máscara para extraer el primer plano de la imagen original
primer_plano = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara_binaria)

# Usar la máscara invertida para extraer el fondo de la imagen de fondo
fondo = cv2.bitwise_and(imagen_fondo, imagen_fondo, mask=mascara_gris)

# Combinar el primer plano y el fondo
resultado = cv2.add(primer_plano, fondo)

# Guardar la imagen resultante
cv2.imwrite('imagen_resultado.jpg', resultado)
