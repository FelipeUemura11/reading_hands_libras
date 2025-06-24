import cv2
import time
import numpy as np
import HandTrackingModule as htm
import tensorflow as tf
import pandas as pd
from datetime import datetime
import os
from collections import deque

##########################################
largura_camera, altura_camera = 1280, 720
##########################################

# tenta diferentes indices de camera
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

detector = htm.DetectorMao()

training_data = []
current_letter = None
collecting_data = False
model = None
scaler_params = None
labels = [chr(i) for i in range(65, 91)]  # A-Z

prediction_buffer = deque(maxlen=5)  # buffer das ultimas 5 predicoes
confidence_threshold = 0.7  # threshold inicial de confianca
min_votes = 3  # minimo de votos para aceitar uma letra
letter_stability_time = 0.8  # tempo minimo para estabilizar uma letra

show_finger_info = True
show_gesture_info = True
show_hand_analysis = True

def normalize_landmarks(landmarks, image_width=640, image_height=480):
    if len(landmarks) < 21:
        return None
    
    # extrair coordenadas do pulso (landmark 0)
    wrist_x, wrist_y = landmarks[0][1], landmarks[0][2]
    
    normalized_features = []
    
    # normalizar todos os landmarks em relacao ao pulso
    for i in range(21):
        if i == 0:  # pulso - usar coordenadas absolutas normalizadas
            x_norm = landmarks[i][1] / image_width
            y_norm = landmarks[i][2] / image_height
        else:  # outros landmarks - coordenadas relativas ao pulso
            x_norm = (landmarks[i][1] - wrist_x) / image_width
            y_norm = (landmarks[i][2] - wrist_y) / image_height
        
        normalized_features.extend([x_norm, y_norm])
    
    return normalized_features

def calcularMaosFeatures(landmarks):
    if len(landmarks) < 21:
        return []
    
    features = []
    wrist = np.array([landmarks[0][1], landmarks[0][2]])
    
    # distancias dos dedos ao pulso
    finger_tips = [4, 8, 12, 16, 20]  # pontas dos dedos
    for tip_idx in finger_tips:
        tip = np.array([landmarks[tip_idx][1], landmarks[tip_idx][2]])
        distance = np.linalg.norm(tip - wrist)
        features.append(distance)
    
    # distancias entre pontas dos dedos
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            tip1 = np.array([landmarks[finger_tips[i]][1], landmarks[finger_tips[i]][2]])
            tip2 = np.array([landmarks[finger_tips[j]][1], landmarks[finger_tips[j]][2]])
            distance = np.linalg.norm(tip1 - tip2)
            features.append(distance)
    
    # angulos entre dedos
    finger_joints = [3, 7, 11, 15, 19]  # juntas dos dedos
    for i in range(len(finger_joints)):
        for j in range(i+1, len(finger_joints)):
            joint1 = np.array([landmarks[finger_joints[i]][1], landmarks[finger_joints[i]][2]])
            joint2 = np.array([landmarks[finger_joints[j]][1], landmarks[finger_joints[j]][2]])
            
            # calcular angulo entre vetores
            v1 = joint1 - wrist
            v2 = joint2 - wrist
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                features.append(angle)
            else:
                features.append(0.0)
    
    return features

def get_votosPedict():
    if len(prediction_buffer) < min_votes:
        return None, 0.0
    
    # contar votos para cada letra
    letter_counts = {}
    total_confidence = {}
    
    for letter, confidence in prediction_buffer:
        if letter not in letter_counts:
            letter_counts[letter] = 0
            total_confidence[letter] = 0.0
        letter_counts[letter] += 1
        total_confidence[letter] += confidence
    
    # encontrar a letra com mais votos
    max_votes = max(letter_counts.values())
    if max_votes < min_votes:
        return None, 0.0
    
    # se ha empate, usar a confianca media
    candidates = [letter for letter, votes in letter_counts.items() if votes == max_votes]
    
    if len(candidates) == 1:
        letter = candidates[0]
        avg_confidence = total_confidence[letter] / letter_counts[letter]
        return letter, avg_confidence
    
    # em caso de empate, escolher a com maior confianca media
    best_letter = None
    best_confidence = 0.0
    
    for letter in candidates:
        avg_confidence = total_confidence[letter] / letter_counts[letter]
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_letter = letter
    
    return best_letter, best_confidence

def sequenciValidaLetras(current_word, new_letter):

    if not current_word:
        return True
    
    # evitar repeticoes excessivas da mesma letra
    if len(current_word) >= 2 and current_word[-1] == current_word[-2] == new_letter:
        return False
    
    # verificar se nao e uma sequencia impossivel (ex: AAA, BBB)
    if len(current_word) >= 3:
        last_three = current_word[-3:] + new_letter
        if len(set(last_three)) == 1:  # todas as letras sao iguais
            return False
    
    return True

def modeloTreinado(data, letter):
    if not os.path.exists('model_hands'):
        os.makedirs('model_hands')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join('model_hands', f"training_data_{letter}_{timestamp}.csv")
    
    # converter landmarks para DataFrame
    df_data = []
    for landmarks, _ in data:
        row = {'letter': letter}
        for i, (_, x, y) in enumerate(landmarks):
            row[f'x_{i}'] = x
            row[f'y_{i}'] = y
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"\nDados salvos em {filename}")
    print(f"Total de amostras: {len(df_data)}")

def load_model():
    try:
        global model, scaler_params
        model_path = os.path.join('model_hands', 'libras_model_improved.h5')
        scaler_path = os.path.join('model_hands', 'scaler_params.npy')
        
        model = tf.keras.models.load_model(model_path)
        scaler_params = np.load(scaler_path, allow_pickle=True).item()
        print("Modelo melhorado carregado com sucesso!")
        return True
    except:
        print("Modelo melhorado nao encontrado na pasta 'model_hands'.")
        print("Tentando carregar modelo antigo...")
        try:
            model = tf.keras.models.load_model('libras_model.h5')
            print("Modelo antigo carregado com sucesso!")
            return True
        except:
            print("Nenhum modelo encontrado. Treine o modelo primeiro.")
            print("Execute: python train_model.py")
            return False

def predictLetra(landmarks):
    if model is None or len(landmarks) != 21:
        return None
    
    normalized_features = normalize_landmarks(landmarks, largura_camera, altura_camera)
    if normalized_features is None:
        return None
    
    additional_features = calcularMaosFeatures(landmarks)
    
    all_features = normalized_features + additional_features
    data = np.array([all_features])
    
    if scaler_params is not None:
        data = (data - scaler_params['mean']) / scaler_params['scale']
    
    predictions = model.predict(data, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    if confidence > confidence_threshold:
        return labels[predicted_class], confidence
    return None

def start_data_collection():
    global collecting_data, current_letter, training_data
    
    print("\n" + "="*50)
    print("INICIANDO COLETA DE DADOS")
    print("="*50)
    
    # fechar temporariamente a janela da camera para permitir input
    cv2.destroyAllWindows()
    time.sleep(0.5)
    
    print("LETRAS DISPONIVEIS: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z")
    print()
    
    while True:
        try:
            letter = input("Digite a letra para coletar dados (A-Z): ").upper().strip()
            if letter in labels:
                current_letter = letter
                collecting_data = True
                training_data = []
                print(f"\nColetando dados para letra {letter}")
                print("Mostre sua mao para a camera e faca o gesto da letra")
                print("Pressione 'c' novamente para parar a coleta")
                print("Pressione 'q' para sair")
                break
            else:
                print(f"Letra invalida! Digite uma letra de A-Z")
                print(f"Letras validas: {', '.join(labels)}")
        except KeyboardInterrupt:
            print("\nOperacao cancelada pelo usuario")
            return
        except Exception as e:
            print(f"Erro no input: {e}")
            print("Tente novamente...")
    
    try:
        cv2.namedWindow("Reconhecimento de Libras", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Reconhecimento de Libras", largura_camera, altura_camera)
    except:
        pass

def stop_data_collection():
    global collecting_data, current_letter, training_data
    
    if collecting_data and training_data:
        print(f"\n" + "="*50)
        print(f"FINALIZANDO COLETA DE DADOS")
        print(f"Letra: {current_letter}")
        print(f"Amostras coletadas: {len(training_data)}")
        print("="*50)
        
        modeloTreinado(training_data, current_letter)
        
        # resetar variaveis
        collecting_data = False
        current_letter = None
        training_data = []
        
        print("Coleta de dados finalizada!")
    else:
        print("Nenhum dado coletado para salvar.")

print("\n=== Reconhecimento de Libras Melhorado ===")
print("Mostre sua mao para a camera para comecar")
print("Pressione 'c' para coletar dados de treinamento")
print("Pressione 'l' para carregar o modelo treinado")
print("Pressione 'f' para alternar informacoes dos dedos")
print("Pressione 'g' para alternar informacoes de gestos")
print("Pressione 'h' para alternar analise da mao")
print("Pressione 'q' para sair")
print("Pressione ' ' (espaco) para limpar a palavra atual")
print("Pressione '+' ou '-' para ajustar a sensibilidade\n")

current_word = ""
last_letter = None
letter_time = 0
letter_delay = letter_stability_time
current_confidence = 0.0

while True:
    sucesso, img = captura.read()
    img = detector.encontrarMaos(img)
    lmList = detector.encontrarPosicao(img, desenhar=False)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    # reconhecimento de letras com sistema de votacao
    if lmList and not collecting_data and model is not None:
        prediction = predictLetra(lmList)
        if prediction:
            letter, confidence = prediction
            prediction_buffer.append((letter, confidence))
            
            voted_letter, voted_confidence = get_votosPedict()
            
            if voted_letter and voted_letter != last_letter and current_time - letter_time > letter_delay:
                if sequenciValidaLetras(current_word, voted_letter):
                    current_word += voted_letter
                    current_confidence = voted_confidence
                    print(f"\rPalavra atual: {current_word} (Confianca: {voted_confidence:.2f})", end="", flush=True)
                    letter_time = current_time
                    last_letter = voted_letter
                else:
                    print(f"\rLetra rejeitada: {voted_letter} (sequencia invalida)", end="", flush=True)

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 180), 3)
    

    if model is not None:
        cv2.putText(img, 'Modelo: Carregado', (40, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(img, 'Modelo: Nao carregado', (40, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.putText(img, f'[c]: Coletar dados', (40, 110), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'[l]: Carregar modelo', (40, 140), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'[f]: Info dedos', (40, 170), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'[g]: Info gestos', (40, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'[h]: Analise mao', (40, 230), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'[ ]: Limpar palavra', (40, 260), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'[++/-]: Sensibilidade', (40, 290), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f'[q]: Sair', (40, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    
    # mostrar configuracoes atuais
    cv2.putText(img, f'Threshold: {confidence_threshold:.2f}', (40, 350), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f'Votos min: {min_votes}', (40, 375), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f'Buffer: {len(prediction_buffer)}', (40, 400), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
    
    if collecting_data:
        cv2.putText(img, f'Coletando dados: {current_letter}', (40, 430), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(img, f'Amostras: {len(training_data)}', (40, 470), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(img, "Pressione 'c' para parar", (40, 510), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
    
    # mostrar palavra atual na tela
    if current_word:
        cv2.putText(img, f'Palavra: {current_word}', (40, 550), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
    
    # mostrar confianca atual
    if current_confidence > 0:
        confidence_color = (0, 255, 0) if current_confidence > 0.8 else (0, 255, 255) if current_confidence > 0.6 else (0, 0, 255)
        cv2.putText(img, f'Confianca: {current_confidence:.2f}', (40, 590), cv2.FONT_HERSHEY_COMPLEX, 0.8, confidence_color, 2)
    
    # informacoes detalhadas dos dedos e gestos
    if lmList and show_finger_info:
        img = detector.desenharInfoDedos(img, lmList)
    
    # informacoes de gestos
    if lmList and show_gesture_info:
        gesture = detector.obterGestoMao(lmList)
        cv2.putText(img, f'Gesto: {gesture}', (largura_camera - 300, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
    
    # analise da mao
    if lmList and show_hand_analysis:
        orientation = detector.obterOrientacaoMao(lmList)
        area = detector.calcularAreaMao(lmList)
        stability = "Estavel" if detector.maoEstaEstavel(lmList) else "Instavel"
        
        cv2.putText(img, f'Orientacao: {orientation}', (largura_camera - 300, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f'Area: {area:.0f}', (largura_camera - 300, 105), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f'Estabilidade: {stability}', (largura_camera - 300, 130), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
    
    if not sucesso:
        print("Erro ao capturar frame!")
        break

    cv2.imshow("Reconhecimento de Libras", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n\nPrograma finalizado!")
        break
    elif key == ord('c'):
        if not collecting_data:
            start_data_collection()
        else:
            stop_data_collection()
    elif key == ord('l'):
        load_model()
    elif key == ord('f'):
        show_finger_info = not show_finger_info
        print(f"\rInformacoes dos dedos: {'Ativadas' if show_finger_info else 'Desativadas'}", end="", flush=True)
    elif key == ord('g'):
        show_gesture_info = not show_gesture_info
        print(f"\rInformacoes de gestos: {'Ativadas' if show_gesture_info else 'Desativadas'}", end="", flush=True)
    elif key == ord('h'):
        show_hand_analysis = not show_hand_analysis
        print(f"\rAnalise da mao: {'Ativada' if show_hand_analysis else 'Desativada'}", end="", flush=True)
    elif key == ord(' '):
        current_word = ""
        last_letter = None
        current_confidence = 0.0
        prediction_buffer.clear()
        print("\rPalavra atual: ", end="", flush=True)
    elif key == ord('+') or key == ord('='):
        confidence_threshold = min(0.95, confidence_threshold + 0.05)
        print(f"\rThreshold ajustado para: {confidence_threshold:.2f}", end="", flush=True)
    elif key == ord('-') or key == ord('_'):
        confidence_threshold = max(0.3, confidence_threshold - 0.05)
        print(f"\rThreshold ajustado para: {confidence_threshold:.2f}", end="", flush=True)

    # coletar dados de treinamento
    if lmList and collecting_data and current_letter:
        training_data.append((lmList, current_letter))
        print(f"\rAmostras coletadas: {len(training_data)}", end="", flush=True)

captura.release()
cv2.destroyAllWindows()