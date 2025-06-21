import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def normalize_landmarks(landmarks, image_width=640, image_height=480):

    if len(landmarks) < 21:
        return None
    
    # Extrair coordenadas do pulso (landmark 0)
    wrist_x, wrist_y = landmarks[0][1], landmarks[0][2]
    
    normalized_features = []
    
    # Normalizar todos os landmarks em relação ao pulso
    for i in range(21):
        if i == 0:  # Pulso - usar coordenadas absolutas normalizadas
            x_norm = landmarks[i][1] / image_width
            y_norm = landmarks[i][2] / image_height
        else:  # Outros landmarks - coordenadas relativas ao pulso
            x_norm = (landmarks[i][1] - wrist_x) / image_width
            y_norm = (landmarks[i][2] - wrist_y) / image_height
        
        normalized_features.extend([x_norm, y_norm])
    
    return normalized_features

def calculate_hand_features(landmarks):

    if len(landmarks) < 21:
        return []
    
    features = []
    wrist = np.array([landmarks[0][1], landmarks[0][2]])
    
    # Distâncias dos dedos ao pulso
    finger_tips = [4, 8, 12, 16, 20]  # Pontas dos dedos
    for tip_idx in finger_tips:
        tip = np.array([landmarks[tip_idx][1], landmarks[tip_idx][2]])
        distance = np.linalg.norm(tip - wrist)
        features.append(distance)
    
    # Distâncias entre pontas dos dedos
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            tip1 = np.array([landmarks[finger_tips[i]][1], landmarks[finger_tips[i]][2]])
            tip2 = np.array([landmarks[finger_tips[j]][1], landmarks[finger_tips[j]][2]])
            distance = np.linalg.norm(tip1 - tip2)
            features.append(distance)
    
    # Ângulos entre dedos
    finger_joints = [3, 7, 11, 15, 19]  # Juntas dos dedos
    for i in range(len(finger_joints)):
        for j in range(i+1, len(finger_joints)):
            joint1 = np.array([landmarks[finger_joints[i]][1], landmarks[finger_joints[i]][2]])
            joint2 = np.array([landmarks[finger_joints[j]][1], landmarks[finger_joints[j]][2]])
            
            # Calcular ângulo entre vetores
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

def load_model_and_data():

    # Carregar modelo
    model_path = os.path.join('model_hands', 'libras_model_improved.h5')
    scaler_path = os.path.join('model_hands', 'scaler_params.npy')
    
    if not os.path.exists(model_path):
        print("Modelo não encontrado! Execute train_model.py primeiro.")
        return None, None, None
    
    model = tf.keras.models.load_model(model_path)
    scaler_params = np.load(scaler_path, allow_pickle=True).item()
    
    print("Modelo carregado com sucesso!")
    
    # Carregar dados
    csv_pattern = os.path.join('model_hands', "training_data_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("Nenhum arquivo CSV encontrado!")
        return None, None, None
    
    all_data = []
    all_labels = []
    all_letters = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        letter = df['letter'].iloc[0]
        
        for _, row in df.iterrows():
            landmarks = []
            for i in range(21):
                x = row[f'x_{i}']
                y = row[f'y_{i}']
                landmarks.append((i, x, y))
            
            # Normalizar landmarks
            normalized_features = normalize_landmarks(landmarks)
            if normalized_features is None:
                continue
            
            # Calcular features adicionais
            additional_features = calculate_hand_features(landmarks)
            
            # Combinar features
            all_features = normalized_features + additional_features
            
            all_data.append(all_features)
            all_labels.append(ord(letter) - ord('A'))
            all_letters.append(letter)
    
    return model, scaler_params, np.array(all_data), np.array(all_labels), all_letters

def evaluate_model_confidence(model, scaler_params, X, y, letters):

    # Aplicar normalização
    X_scaled = (X - scaler_params['mean']) / scaler_params['scale']
    
    # Fazer predições
    predictions = model.predict(X_scaled, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    # Calcular accuracy geral
    overall_accuracy = accuracy_score(y, predicted_classes)
    
    # Avaliar diferentes thresholds de confiança
    thresholds = np.arange(0.3, 1.0, 0.05)
    threshold_results = []
    
    for threshold in thresholds:
        # Filtrar predições com confiança acima do threshold
        high_confidence_mask = confidence_scores >= threshold
        if np.sum(high_confidence_mask) == 0:
            continue
        
        filtered_y = y[high_confidence_mask]
        filtered_pred = predicted_classes[high_confidence_mask]
        filtered_conf = confidence_scores[high_confidence_mask]
        
        # Calcular métricas
        accuracy = accuracy_score(filtered_y, filtered_pred)
        coverage = np.sum(high_confidence_mask) / len(y)
        avg_confidence = np.mean(filtered_conf)
        
        threshold_results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'coverage': coverage,
            'avg_confidence': avg_confidence,
            'samples': np.sum(high_confidence_mask)
        })
    
    return threshold_results, predictions, predicted_classes, confidence_scores

def analyze_per_letter_confidence(model, scaler_params, X, y, letters):
    X_scaled = (X - scaler_params['mean']) / scaler_params['scale']
    predictions = model.predict(X_scaled, verbose=0)
    
    letter_analysis = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    
    for i, (true_label, pred_probs) in enumerate(zip(y, predictions)):
        letter = chr(true_label + ord('A'))
        predicted_class = np.argmax(pred_probs)
        confidence = np.max(pred_probs)
        is_correct = (true_label == predicted_class)
        
        letter_analysis[letter]['total'] += 1
        letter_analysis[letter]['confidences'].append(confidence)
        
        if is_correct:
            letter_analysis[letter]['correct'] += 1
    
    # Calcular métricas por letra
    letter_metrics = {}
    for letter, data in letter_analysis.items():
        accuracy = data['correct'] / data['total']
        avg_confidence = np.mean(data['confidences'])
        std_confidence = np.std(data['confidences'])
        
        letter_metrics[letter] = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence,
            'samples': data['total']
        }
    
    return letter_metrics

def plot_confidence_analysis(threshold_results, letter_metrics):
    plt.figure(figsize=(20, 12))
    # Gráfico 1: Threshold vs Accuracy/Coverage
    plt.subplot(2, 3, 1)
    thresholds = [r['threshold'] for r in threshold_results]
    accuracies = [r['accuracy'] for r in threshold_results]
    coverages = [r['coverage'] for r in threshold_results]
    
    plt.plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2)
    plt.plot(thresholds, coverages, 'r--', label='Coverage', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Accuracy vs Coverage por Threshold')
    plt.legend()
    plt.grid(True)
    
    # Gráfico 2: Accuracy por letra
    plt.subplot(2, 3, 2)
    letters = list(letter_metrics.keys())
    accuracies = [letter_metrics[letter]['accuracy'] for letter in letters]
    
    colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in accuracies]
    bars = plt.bar(letters, accuracies, color=colors)
    plt.xlabel('Letra')
    plt.ylabel('Accuracy')
    plt.title('Accuracy por Letra')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.2f}', ha='center', va='bottom')
    
    # Gráfico 3: Confiança média por letra
    plt.subplot(2, 3, 3)
    avg_confidences = [letter_metrics[letter]['avg_confidence'] for letter in letters]
    std_confidences = [letter_metrics[letter]['std_confidence'] for letter in letters]
    
    plt.errorbar(letters, avg_confidences, yerr=std_confidences, fmt='o-', capsize=5)
    plt.xlabel('Letra')
    plt.ylabel('Confiança Média')
    plt.title('Confiança Média por Letra')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Gráfico 4: Número de amostras por letra
    plt.subplot(2, 3, 4)
    samples = [letter_metrics[letter]['samples'] for letter in letters]
    plt.bar(letters, samples, color='skyblue')
    plt.xlabel('Letra')
    plt.ylabel('Número de Amostras')
    plt.title('Distribuição de Amostras por Letra')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Gráfico 5: Threshold vs Número de Amostras
    plt.subplot(2, 3, 5)
    samples_at_threshold = [r['samples'] for r in threshold_results]
    plt.plot(thresholds, samples_at_threshold, 'g-', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Número de Amostras')
    plt.title('Amostras por Threshold')
    plt.grid(True)
    
    # Gráfico 6: Confiança vs Accuracy (scatter)
    plt.subplot(2, 3, 6)
    confidences = [letter_metrics[letter]['avg_confidence'] for letter in letters]
    plt.scatter(confidences, accuracies, s=100, alpha=0.7)
    
    # Adicionar labels das letras
    for i, letter in enumerate(letters):
        plt.annotate(letter, (confidences[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Confiança Média')
    plt.ylabel('Accuracy')
    plt.title('Confiança vs Accuracy por Letra')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_hands/confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_confidence_report(threshold_results, letter_metrics):

    print("\n" + "="*60)
    print("RELATÓRIO DE CONFIANÇA DO MODELO")
    print("="*60)
    
    # Resumo geral
    print("\nRESUMO GERAL:")
    print(f"Total de letras analisadas: {len(letter_metrics)}")
    
    # Melhores e piores letras
    sorted_letters = sorted(letter_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\nTOP 5 MELHORES LETRAS:")
    for i, (letter, metrics) in enumerate(sorted_letters[:5]):
        print(f"{i+1}. {letter}: {metrics['accuracy']:.3f} (conf: {metrics['avg_confidence']:.3f})")
    
    print(f"\nTOP 5 PIORES LETRAS:")
    for i, (letter, metrics) in enumerate(sorted_letters[-5:]):
        print(f"{i+1}. {letter}: {metrics['accuracy']:.3f} (conf: {metrics['avg_confidence']:.3f})")
    
    # Análise de threshold
    print(f"\nANÁLISE DE THRESHOLD:")
    print("Threshold | Accuracy | Coverage | Amostras")
    print("-" * 45)
    
    for result in threshold_results:
        print(f"{result['threshold']:.2f}     | {result['accuracy']:.3f}    | {result['coverage']:.3f}     | {result['samples']}")
    
    # Recomendações
    print(f"\nRECOMENDAÇÕES:")
    
    # Encontrar threshold ótimo (balance entre accuracy e coverage)
    optimal_threshold = None
    best_score = 0
    
    for result in threshold_results:
        # Score = accuracy * coverage (balance entre precisão e cobertura)
        score = result['accuracy'] * result['coverage']
        if score > best_score:
            best_score = score
            optimal_threshold = result['threshold']
    
    print(f"• Threshold ótimo recomendado: {optimal_threshold:.2f}")
    print(f"• Este threshold oferece o melhor balance entre precisão e cobertura")
    
    # Letras que precisam de mais dados
    low_accuracy_letters = [letter for letter, metrics in letter_metrics.items() 
                           if metrics['accuracy'] < 0.7]
    if low_accuracy_letters:
        print(f"• Letras que precisam de mais dados de treinamento: {', '.join(low_accuracy_letters)}")
    
    # Salvar relatório em arquivo
    with open('model_hands/confidence_report.txt', 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE CONFIANÇA DO MODELO\n")
        f.write("="*60 + "\n")
        f.write(f"Data: {pd.Timestamp.now()}\n\n")
        
        f.write("RESUMO GERAL:\n")
        f.write(f"Total de letras analisadas: {len(letter_metrics)}\n\n")
        
        f.write("MÉTRICAS POR LETRA:\n")
        f.write("Letra | Accuracy | Confiança Média | Amostras\n")
        f.write("-" * 50 + "\n")
        for letter in sorted(letter_metrics.keys()):
            metrics = letter_metrics[letter]
            f.write(f"{letter}     | {metrics['accuracy']:.3f}    | {metrics['avg_confidence']:.3f}        | {metrics['samples']}\n")
        
        f.write(f"\nTHRESHOLD ÓTIMO: {optimal_threshold:.2f}\n")
    
    print(f"\nRelatório salvo em: model_hands/confidence_report.txt")

def main():
    print("=== Validação e Análise de Confiança do Modelo ===")
    
    # Carregar modelo e dados
    result = load_model_and_data()
    if result is None:
        return
    
    model, scaler_params, X, y, letters = result
    
    print(f"Dados carregados: {len(X)} amostras")
    print(f"Letras únicas: {sorted(set(letters))}")
    
    # Avaliar confiança
    print("\nAvaliando confiança do modelo...")
    threshold_results, predictions, predicted_classes, confidence_scores = evaluate_model_confidence(
        model, scaler_params, X, y, letters
    )
    
    # Analisar por letra
    print("Analisando confiança por letra...")
    letter_metrics = analyze_per_letter_confidence(model, scaler_params, X, y, letters)
    
    # Gerar relatório
    generate_confidence_report(threshold_results, letter_metrics)
    
    # Plotar gráficos
    print("\nGerando gráficos de análise...")
    plot_confidence_analysis(threshold_results, letter_metrics)
    
    print("\nAnálise concluída!")
    print("Arquivos gerados:")
    print("• model_hands/confidence_analysis.png")
    print("• model_hands/confidence_report.txt")

if __name__ == "__main__":
    main() 