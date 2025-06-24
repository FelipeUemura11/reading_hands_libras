import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

class DetectorMao:
    def __init__(self, modo=False, maxMaos=2, confiancaDetecao=0.7, confiancaRastreamento=0.7):
        self.modo = modo
        self.maxMaos = maxMaos
        self.confiancaDetecao = confiancaDetecao
        self.confiancaRastreamento = confiancaRastreamento

        self.mpMaos = mp.solutions.hands
        self.maos = self.mpMaos.Hands(
            static_image_mode=self.modo,
            max_num_hands=self.maxMaos,
            model_complexity=1,
            min_detection_confidence=self.confiancaDetecao,
            min_tracking_confidence=self.confiancaRastreamento
        )
        self.mpDesenho = mp.solutions.drawing_utils
        
        # filtros de suavizacao para reduzir tremores
        self.filtrosLandmarks = {}
        self.tamanhoJanelaFiltro = 5
        
        # pontas dos dedos (MediaPipe)
        self.idsPontasDedos = [4, 8, 12, 16, 20]  # polegar, indicador, medio, anelar, minimo
        
        # pontos de referencia para deteccao de dedos
        self.pontosReferenciaDedos = {
            4: 3,   # polegar: ponta vs base
            8: 6,   # indicador: ponta vs base
            12: 10, # medio: ponta vs base
            16: 14, # anelar: ponta vs base
            20: 18  # minimo: ponta vs base
        }
        
        # historico para estabilizacao
        self.historicoMao = deque(maxlen=10)
        self.limiarEstabilidade = 0.02

    def encontrarMaos(self, imagem, desenhar=True):
        imagemRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        self.resultados = self.maos.process(imagemRGB)
        
        if self.resultados.multi_hand_landmarks:
            for landmarksMao in self.resultados.multi_hand_landmarks:
                if desenhar:
                    self.mpDesenho.draw_landmarks(imagem, landmarksMao, self.mpMaos.HAND_CONNECTIONS)
        return imagem

    def encontrarPosicao(self, imagem, numeroMao=0, desenhar=True):
        listaLandmarks = []
        if self.resultados.multi_hand_landmarks:
            if len(self.resultados.multi_hand_landmarks) > numeroMao:
                minhaMao = self.resultados.multi_hand_landmarks[numeroMao]
                altura, largura, canais = imagem.shape
                
                for id, landmark in enumerate(minhaMao.landmark):
                    posX, posY = int(landmark.x * largura), int(landmark.y * altura)
                    
                    # Aplicar filtro de suavizacao
                    posicaoSuavizada = self.aplicarFiltroSuavizacao(id, (posX, posY))
                    listaLandmarks.append((id, posicaoSuavizada[0], posicaoSuavizada[1]))
                    
                    if desenhar:
                        # Desenhar pontos com cores diferentes para diferentes tipos
                        if id in self.idsPontasDedos:
                            cv2.circle(imagem, (posicaoSuavizada[0], posicaoSuavizada[1]), 12, (0, 255, 0), cv2.FILLED)
                        elif id in [0, 5, 9, 13, 17]:  # Pontos base dos dedos
                            cv2.circle(imagem, (posicaoSuavizada[0], posicaoSuavizada[1]), 8, (255, 0, 0), cv2.FILLED)
                        else:
                            cv2.circle(imagem, (posicaoSuavizada[0], posicaoSuavizada[1]), 6, (255, 0, 255), cv2.FILLED)
        return listaLandmarks

    def aplicarFiltroSuavizacao(self, idLandmark, posicao):
        # aplica filtro de media movel para suavizar os landmarks
        if idLandmark not in self.filtrosLandmarks:
            self.filtrosLandmarks[idLandmark] = deque(maxlen=self.tamanhoJanelaFiltro)
        
        self.filtrosLandmarks[idLandmark].append(posicao)
        
        if len(self.filtrosLandmarks[idLandmark]) < 2:
            return posicao
        
        # calcular media ponderada (mais peso para posicoes recentes)
        pesos = np.linspace(0.5, 1.0, len(self.filtrosLandmarks[idLandmark]))
        pesos = pesos / np.sum(pesos)
        
        mediaX = int(np.average([pos[0] for pos in self.filtrosLandmarks[idLandmark]], weights=pesos))
        mediaY = int(np.average([pos[1] for pos in self.filtrosLandmarks[idLandmark]], weights=pesos))
        
        return (mediaX, mediaY)

    def detectarDedos(self, listaLandmarks):
        dedos = []
        
        if len(listaLandmarks) < 21:
            return dedos
        
        # Deteccao especial para o polegar (horizontal vs vertical)
        pontaPolegar = listaLandmarks[4]
        basePolegar = listaLandmarks[3]
        juntaPolegar = listaLandmarks[2]
        
        # Calcular orientacao do polegar
        anguloPolegar = self.calcularAnguloDedo(pontaPolegar, basePolegar, juntaPolegar)
        polegarEstendido = anguloPolegar > 160  # Angulo mais permissivo para polegar
        
        # Para outros dedos, usar comparacao vertical
        for i in range(1, 5):  # Indicador, Medio, Anelar, Minimo
            ponta = listaLandmarks[self.idsPontasDedos[i]]
            primeiraJunta = listaLandmarks[self.idsPontasDedos[i] - 1]  # Primeira junta
            baseDedo = listaLandmarks[self.idsPontasDedos[i] - 2]  # Base do dedo
            
            # Calcular angulo do dedo
            anguloDedo = self.calcularAnguloDedo(ponta, primeiraJunta, baseDedo)
            dedoEstendido = anguloDedo > 150  # Angulo para outros dedos
            
            dedos.append(1 if dedoEstendido else 0)
        
        # Adicionar polegar no inicio
        dedos.insert(0, 1 if polegarEstendido else 0)
        
        return dedos

    def calcularAnguloDedo(self, ponta, junta, base):
        # Vetores do dedo
        vetor1 = np.array([junta[1] - base[1], junta[2] - base[2]])
        vetor2 = np.array([ponta[1] - junta[1], ponta[2] - junta[2]])
        
        # Calcular angulo
        cosAngulo = np.dot(vetor1, vetor2) / (np.linalg.norm(vetor1) * np.linalg.norm(vetor2))
        cosAngulo = np.clip(cosAngulo, -1.0, 1.0)
        angulo = np.degrees(np.arccos(cosAngulo))
        
        return angulo

    def obterOrientacaoMao(self, listaLandmarks):
        if len(listaLandmarks) < 21:
            return "desconhecida"
        
        # Usar pontos da palma para determinar orientacao
        pontosPalma = [listaLandmarks[0], listaLandmarks[5], listaLandmarks[9], listaLandmarks[13], listaLandmarks[17]]
        
        # Calcular centro da palma
        centroPalmaX = sum(ponto[1] for ponto in pontosPalma) / len(pontosPalma)
        centroPalmaY = sum(ponto[2] for ponto in pontosPalma) / len(pontosPalma)
        
        # Comparar posicao do pulso com centro da palma
        pulso = listaLandmarks[0]
        
        if pulso[2] < centroPalmaY:
            return "palma_cima"
        else:
            return "palma_baixo"

    def calcularAreaMao(self, listaLandmarks):
        if len(listaLandmarks) < 21:
            return 0
        
        # Pontos do contorno da mao
        contornoMao = [
            listaLandmarks[0], listaLandmarks[1], listaLandmarks[2], listaLandmarks[3], listaLandmarks[4],  # Polegar
            listaLandmarks[5], listaLandmarks[6], listaLandmarks[7], listaLandmarks[8],  # Indicador
            listaLandmarks[9], listaLandmarks[10], listaLandmarks[11], listaLandmarks[12],  # Medio
            listaLandmarks[13], listaLandmarks[14], listaLandmarks[15], listaLandmarks[16],  # Anelar
            listaLandmarks[17], listaLandmarks[18], listaLandmarks[19], listaLandmarks[20]   # Minimo
        ]
        
        # Calcular area usando formula do poligono
        area = 0
        n = len(contornoMao)
        for i in range(n):
            j = (i + 1) % n
            area += contornoMao[i][1] * contornoMao[j][2]
            area -= contornoMao[j][1] * contornoMao[i][2]
        area = abs(area) / 2
        
        return area

    def maoEstaEstavel(self, listaLandmarks):
        if len(listaLandmarks) < 21:
            return False
        
        # Adicionar posicao atual ao historico
        maoAtual = np.array([(landmark[1], landmark[2]) for landmark in listaLandmarks])
        self.historicoMao.append(maoAtual)
        
        if len(self.historicoMao) < 3:
            return False
        
        # Calcular variacao media das posicoes
        variacoes = []
        for i in range(len(self.historicoMao) - 1):
            diferenca = np.linalg.norm(self.historicoMao[i+1] - self.historicoMao[i])
            variacoes.append(diferenca)
        
        variacaoMedia = np.mean(variacoes)
        return variacaoMedia < self.limiarEstabilidade

    def obterDistanciasDedos(self, listaLandmarks):
        if len(listaLandmarks) < 21:
            return {}
        
        distancias = {}
        pontasDedos = [4, 8, 12, 16, 20]
        
        for i in range(len(pontasDedos)):
            for j in range(i+1, len(pontasDedos)):
                ponta1 = listaLandmarks[pontasDedos[i]]
                ponta2 = listaLandmarks[pontasDedos[j]]
                
                distancia = math.sqrt((ponta1[1] - ponta2[1])**2 + (ponta1[2] - ponta2[2])**2)
                chave = f"d_{pontasDedos[i]}_{pontasDedos[j]}"
                distancias[chave] = distancia
        
        return distancias

    def obterGestoMao(self, listaLandmarks):
        if len(listaLandmarks) < 21:
            return "desconhecido"
        
        dedos = self.detectarDedos(listaLandmarks)
        
        # gestos padrão de um humano
        if dedos == [0, 1, 1, 1, 1]:  # Apenas polegar fechado
            return "A"
        elif dedos == [1, 0, 0, 0, 0]:  # Apenas polegar
            return "polegar_cima"
        elif dedos == [0, 0, 0, 0, 0]:  # Punho fechado
            return "punho"
        elif dedos == [1, 1, 1, 1, 1]:  # Mao aberta
            return "mao_aberta"
        elif dedos == [0, 1, 0, 0, 0]:  # Apenas indicador
            return "apontando"
        elif dedos == [0, 1, 1, 0, 0]:  # Paz
            return "paz"
        elif dedos == [1, 1, 0, 0, 0]:  # Pistola
            return "pistola"
        else:
            return "personalizado"

    def desenharInfoDedos(self, imagem, listaLandmarks, dedos=None):
        if len(listaLandmarks) < 21:
            return imagem
        
        if dedos is None:
            dedos = self.detectarDedos(listaLandmarks)
        
        nomesDedos = ["Polegar", "Indicador", "Medio", "Anelar", "Minimo"]
        
        # Desenhar informacoes para cada dedo
        for i, (idDedo, nome) in enumerate(zip(self.idsPontasDedos, nomesDedos)):
            ponta = listaLandmarks[idDedo]
            status = "Estendido" if dedos[i] else "Dobrado"
            cor = (0, 255, 0) if dedos[i] else (0, 0, 255)
            
            cv2.putText(imagem, f"{nome}: {status}", 
                       (ponta[1] + 15, ponta[2] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
            
            # Desenhar circulo colorido na ponta
            cv2.circle(imagem, (ponta[1], ponta[2]), 8, cor, cv2.FILLED)
        
        # Mostrar orientacao da mao
        orientacao = self.obterOrientacaoMao(listaLandmarks)
        cv2.putText(imagem, f"Orientacao: {orientacao}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar estabilidade
        estabilidade = "Estavel" if self.maoEstaEstavel(listaLandmarks) else "Instavel"
        corEstabilidade = (0, 255, 0) if self.maoEstaEstavel(listaLandmarks) else (0, 0, 255)
        cv2.putText(imagem, f"Estabilidade: {estabilidade}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, corEstabilidade, 2)
        
        return imagem

class handDetector(DetectorMao):
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        super().__init__(mode, maxHands, detectionCon, trackCon)
    
    def findHands(self, img, draw=True):
        return self.encontrarMaos(img, draw)
    
    def findPosition(self, img, handNo=0, draw=True):
        return self.encontrarPosicao(img, handNo, draw)
