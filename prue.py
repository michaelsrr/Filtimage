import cv2

def overlay_images(person_image_path, background_image_path, output_path):
    # Cargar las imágenes
    person_image = cv2.imread(person_image_path)
    background_image = cv2.imread(background_image_path)

    # Verificar si las imágenes se cargaron correctamente
    if person_image is None or background_image is None:
        print("Error al cargar las imágenes.")
        return

    # Inicializar el clasificador Haarcascades para la detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convertir la imagen de la persona a escala de grises
    gray_person = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen de la persona
    faces = face_cascade.detectMultiScale(gray_person, scaleFactor=1.3, minNeighbors=5)

    # Tomar solo el primer rostro detectado (asumimos que hay solo una persona en la imagen)
    if len(faces) > 0:
        x, y, w, h = faces[0]

        # Extraer la región de interés (ROI) que contiene la persona
        person_roi = person_image[y:y + h, x:x + w]

        # Redimensionar la imagen de fondo para que coincida con la ROI de la persona
        background_image_resized = cv2.resize(background_image, (w, h))

        # Superponer la persona en el fondo
        result = cv2.addWeighted(person_roi, 1, background_image_resized, 1, 0)

        # Actualizar la imagen de fondo con la superposición
        background_image[y:y + h, x:x + w] = result

        # Guardar la imagen resultante
        cv2.imwrite(output_path, background_image)
        print("Imagen resultante guardada en:", output_path)
    else:
        print("No se detectaron rostros en la imagen de la persona.")

# Rutas de las imágenes de entrada y salida
person_image_path = "C:/Users/Michael/Documents/GitHub/Filtimage/imagen.jpg"
background_image_path = "C:/Users/Michael/Documents/GitHub/Filtimage/imagen_fondo.jpeg"
output_path = "C:/Users/Michael/Documents/GitHub/Filtimage/imagen_resultante.jpg"

# Llamar a la función para superponer las imágenes
overlay_images(person_image_path, background_image_path, output_path)
