import numpy as np
import cv2                              # Importamos librerias

#Vamos a capturar el objeto que queremos identificar

cap = cv2.VideoCapture(1)              # Elegimos la camara con la que vamos a hacer la deteccion
while(True):
    ret,frame = cap.read()             # Leemos el video
    cv2.imshow('Objeto',frame)         # Mostramos el video en pantalla
    if cv2.waitKey(1) == 27:           # Cuando oprimamos "Escape" rompe el video
        break
cv2.imwrite('objeto.jpg',frame)       # Guardamos la ultima caputra del video como imagen
cap.release()                         # Cerramos
cv2.destroyAllWindows()

#Leemos la imagen del objeto que queremos identificar
obj = cv2.imread('objeto.jpg',0)      # Leemos la imagen
recorte = obj[160:300, 230:380]       # Recortamos la imagen para que quede solo el objeto (fila:fila, colum:colum)
cv2.imshow('objeto',recorte)          # Mostramos en pantalla el objeto a reconocer

#Una vez tenemos el objeto definido tomamos la foto con el resto de objetos
cap = cv2.VideoCapture(1)                 # Elegimos la camara con la que vamos a hacer la deteccion
while(True):
    ret2,frame2 = cap.read()              # Leemos el video
    cv2.imshow('Deteccion',frame2)        # Mostramos el video en pantalla
    if cv2.waitKey(1) == 27:              # Cuando oprimamos "Escape" rompe el video
        break
cv2.imwrite('Deteccion.jpg',frame2)       # Guardamos la ultima caputra del video como imagen
cap.release()                             # Cerramos
cv2.destroyAllWindows()

#Mostramos la imagen con todos los objetos
img = cv2.imread('Deteccion.jpg',3)
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Pasamos la imagen a escala de grises
cv2.imshow('Deteccion',img)

#Empezamos el algoritmo
w, h = recorte.shape[::-1]                                           # Extraemos el ancho y el alto del recorte del objeto
deteccion  = cv2.matchTemplate(gris, recorte, cv2.TM_CCOEFF_NORMED)  # Realizamos la deteccion por patrones
umbral = 0.75                                                        # Asignamos un umbral para filtrar objetos parecidos
ubi = np.where(deteccion >= umbral)                                  # La ubicacion de los objetos la vamos a guardar cuando supere el umbral
for pt in zip (*ubi[::-1]):                                          # Creamos un for para dibujar todos los rectangulos
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (255,0,0), 1)         # Dibujamos los n rectangulos que hayamos identificado con el tamaño del recorte y de color

cv2.imshow('Deteccion',img)
