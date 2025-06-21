import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

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

def augment_data(landmarks, label, num_augmentations=5):
    augmented_data = []
    
    for _ in range(num_augmentations):
        # Adicionar ruído gaussiano
        noise_factor = np.random.uniform(0.01, 0.03)
        augmented_landmarks = []
        
        for landmark in landmarks:
            x_noise = np.random.normal(0, noise_factor)
            y_noise = np.random.normal(0, noise_factor)
            augmented_landmarks.append((landmark[0], landmark[1] + x_noise, landmark[2] + y_noise))
        
        augmented_data.append((augmented_landmarks, label))
    
    # Adicionar variações de escala
    for scale_factor in [0.95, 1.05]:
        scaled_landmarks = []
        wrist_x, wrist_y = landmarks[0][1], landmarks[0][2]
        
        for landmark in landmarks:
            if landmark[0] == 0:  # Pulso - não escalar
                scaled_landmarks.append(landmark)
            else:
                # Escalar em relação ao pulso
                dx = (landmark[1] - wrist_x) * scale_factor
                dy = (landmark[2] - wrist_y) * scale_factor
                scaled_landmarks.append((landmark[0], wrist_x + dx, wrist_y + dy))
        
        augmented_data.append((scaled_landmarks, label))
    
    # Adicionar rotação leve
    for angle in [-5, 5]:  # Rotação de ±5 graus
        rotated_landmarks = []
        wrist_x, wrist_y = landmarks[0][1], landmarks[0][2]
        angle_rad = np.radians(angle)
        
        for landmark in landmarks:
            if landmark[0] == 0:  # Pulso - não rotacionar
                rotated_landmarks.append(landmark)
            else:
                # Rotacionar em relação ao pulso
                dx = landmark[1] - wrist_x
                dy = landmark[2] - wrist_y
                
                new_x = dx * np.cos(angle_rad) - dy * np.sin(angle_rad) + wrist_x
                new_y = dx * np.sin(angle_rad) + dy * np.cos(angle_rad) + wrist_y
                
                rotated_landmarks.append((landmark[0], new_x, new_y))
        
        augmented_data.append((rotated_landmarks, label))
    
    return augmented_data

def load_training_data():
    all_data = []
    all_labels = []
    
    if not os.path.exists('model_hands'):
        print("Pasta 'model_hands' não encontrada!")
        print("Execute primeiro a coleta de dados usando main.py")
        return None, None, None
    
    csv_pattern = os.path.join('model_hands', "training_data_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("Nenhum arquivo CSV encontrado na pasta 'model_hands'!")
        print("Execute primeiro a coleta de dados usando main.py")
        return None, None, None
    
    print(f"Encontrados {len(csv_files)} arquivos de dados")
    
    # Contar amostras por letra
    letter_counts = {}
    
    for csv_file in csv_files:
        print(f"Carregando dados de {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        letter = df['letter'].iloc[0]
        
        if letter not in letter_counts:
            letter_counts[letter] = 0
        
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
            all_labels.append(ord(letter) - ord('A'))  # Converter letras para números (0-25)
            letter_counts[letter] += 1
            
            # Data augmentation mais robusto
            augmented_samples = augment_data(landmarks, ord(letter) - ord('A'))
            for aug_landmarks, aug_label in augmented_samples:
                aug_normalized = normalize_landmarks(aug_landmarks)
                if aug_normalized is not None:
                    aug_additional = calculate_hand_features(aug_landmarks)
                    aug_features = aug_normalized + aug_additional
                    all_data.append(aug_features)
                    all_labels.append(aug_label)
                    letter_counts[letter] += 1
    
    if not all_data:
        print("Nenhum dado de treinamento válido encontrado!")
        return None, None, None
    
    print("\nDistribuição de amostras por letra:")
    for letter in sorted(letter_counts.keys()):
        print(f"{letter}: {letter_counts[letter]} amostras")
        
    return np.array(all_data), np.array(all_labels), len(all_data[0])

def create_improved_model(input_shape, num_classes=26):

    model = tf.keras.Sequential([
        # Camada de entrada
        tf.keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Camadas intermediárias
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Camada de saída
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar com learning rate fixo (removido o schedule para evitar conflito)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model_with_cross_validation():
    print("Carregando dados de treinamento...")
    X, y, input_shape = load_training_data()
    
    if X is None or y is None:
        return
    
    print(f"Dados carregados: {len(X)} amostras")
    print(f"Shape dos dados: {X.shape}")
    print(f"Classes únicas: {np.unique(y)}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # CROSS VALIDATION
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    fold_histories = []
    
    print(f"\nIniciando Cross Validation com {n_splits} folds...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled), 1):
        print(f"\n>> Fold {fold}/{n_splits} <<")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = create_improved_model(input_shape)
        
        # Callbacks para melhor treinamento
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        )
        
        # Treinar modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Avaliar modelo
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_scores.append(val_accuracy)
        fold_histories.append(history)
        
        print(f"Fold {fold} - Validação Accuracy: {val_accuracy:.4f}")
    
    # Resultados finais
    print(f"\n>> RESULTADOS FINAIS <<")
    print(f"Accuracy média: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"Melhor fold: {np.argmax(fold_scores) + 1} ({max(fold_scores):.4f})")
    
    # Treinar modelo final com todos os dados
    print("\nTreinando modelo final com todos os dados...")
    final_model = create_improved_model(input_shape)
    
    # Callbacks para modelo final
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=20,
        restore_best_weights=True
    )
    
    # Treinar modelo final
    final_history = final_model.fit(
        X_scaled, y,
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Salvar modelo e parâmetros
    if not os.path.exists('model_hands'):
        os.makedirs('model_hands')
    
    model_path = os.path.join('model_hands', 'libras_model_improved.h5')
    scaler_path = os.path.join('model_hands', 'scaler_params.npy')
    
    final_model.save(model_path)
    np.save(scaler_path, {
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })
    
    print(f"\nModelo salvo em: {model_path}")
    print(f"Parâmetros do scaler salvos em: {scaler_path}")
    
    # Avaliação final
    final_accuracy = final_model.evaluate(X_scaled, y, verbose=0)[1]
    print(f"Accuracy final: {final_accuracy:.4f}")
    
    # Plotar gráficos de treinamento
    plot_training_history(final_history, fold_histories)

def plot_training_history(final_history, fold_histories):
    """
    Plota gráficos do histórico de treinamento
    """
    try:
        plt.figure(figsize=(15, 5))
        
        # Gráfico do modelo final
        plt.subplot(1, 3, 1)
        plt.plot(final_history.history['accuracy'], label='Treino')
        plt.title('Modelo Final - Accuracy')
        plt.xlabel('Época')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(final_history.history['loss'], label='Treino')
        plt.title('Modelo Final - Loss')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Gráfico dos folds
        plt.subplot(1, 3, 3)
        fold_accuracies = [h.history['val_accuracy'][-1] for h in fold_histories]
        plt.bar(range(1, len(fold_accuracies) + 1), fold_accuracies)
        plt.title('Accuracy por Fold')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('model_hands/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gráficos salvos em: model_hands/training_history.png")
        
    except Exception as e:
        print(f"Erro ao plotar gráficos: {e}")

if __name__ == "__main__":
    print(" >> Treinamento do Modelo de Libras Melhorado <<")
    train_model_with_cross_validation() 