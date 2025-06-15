import cv2
import time
import numpy as np
import HandTrackingModule as htm

##########################################
largura_camera, altura_camera = 640, 480
##########################################

# Tenta diferentes índices de câmera
for camera_index in [0, 1, 2]:
    captura = cv2.VideoCapture(camera_index)
    if captura.isOpened():
        print(f"Câmera encontrada no índice {camera_index}")
        break
else:
    print("Nenhuma câmera encontrada! Verifique se sua câmera está conectada.")
    exit()

captura.set(3, largura_camera)
captura.set(4, altura_camera)
previous_time = 0

detector = htm.handDetector()


while True:
    sucesso, img = captura.read()
    img = detector.findHands(img)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 180), 3)
    
    if not sucesso:
        print("Erro ao capturar frame!")
        break

    cv2.imshow("Img", img)
    
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
captura.release()
cv2.destroyAllWindows()