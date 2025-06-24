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

def calculate_hand_features(landmarks):
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

def augment_data(landmarks, label, num_augmentations=5):
    augmented_data = []
    
    for _ in range(num_augmentations):
        # adicionar ruido gaussiano
        noise_factor = np.random.uniform(0.01, 0.03)
        augmented_landmarks = []
        
        for landmark in landmarks:
            x_noise = np.random.normal(0, noise_factor)
            y_noise = np.random.normal(0, noise_factor)
            augmented_landmarks.append((landmark[0], landmark[1] + x_noise, landmark[2] + y_noise))
        
        augmented_data.append((augmented_landmarks, label))
    
    # adicionar variacoes de escala
    for scale_factor in [0.95, 1.05]:
        scaled_landmarks = []
        wrist_x, wrist_y = landmarks[0][1], landmarks[0][2]
        
        for landmark in landmarks:
            if landmark[0] == 0:  # pulso - nao escalar
                scaled_landmarks.append(landmark)
            else:
                # escalar em relacao ao pulso
                dx = (landmark[1] - wrist_x) * scale_factor
                dy = (landmark[2] - wrist_y) * scale_factor
                scaled_landmarks.append((landmark[0], wrist_x + dx, wrist_y + dy))
        
        augmented_data.append((scaled_landmarks, label))
    
    # adicionar rotacao leve
    for angle in [-5, 5]:  # rotacao de ±5 graus
        rotated_landmarks = []
        wrist_x, wrist_y = landmarks[0][1], landmarks[0][2]
        angle_rad = np.radians(angle)
        
        for landmark in landmarks:
            if landmark[0] == 0:  # pulso - nao rotacionar
                rotated_landmarks.append(landmark)
            else:
                # rotacionar em relacao ao pulso
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
        print("Pasta 'model_hands' nao encontrada!")
        print("Execute primeiro a coleta de dados usando main.py")
        return None, None, None
    
    csv_pattern = os.path.join('model_hands', "training_data_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("Nenhum arquivo CSV encontrado na pasta 'model_hands'!")
        print("Execute primeiro a coleta de dados usando main.py")
        return None, None, None
    
    print(f"Encontrados {len(csv_files)} arquivos de dados")
    
    # contar amostras por letra
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
            
            # normalizar landmarks
            normalized_features = normalize_landmarks(landmarks)
            if normalized_features is None:
                continue
            
            # calcular features adicionais
            additional_features = calculate_hand_features(landmarks)
            
            # combinar features
            all_features = normalized_features + additional_features
            
            all_data.append(all_features)
            all_labels.append(ord(letter) - ord('A'))  # converter letras para numeros (0-25)
            letter_counts[letter] += 1
            
            # data augmentation mais robusto
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
        print("Nenhum dado de treinamento valido encontrado!")
        return None, None, None
    
    print("\nDistribuicao de amostras por letra:")
    for letter in sorted(letter_counts.keys()):
        print(f"{letter}: {letter_counts[letter]} amostras")
    
    # converter para arrays numpy
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # salvar parametros do scaler
    scaler_params = {
        'mean': scaler.mean_,
        'scale': scaler.scale_
    }
    
    if not os.path.exists('model_hands'):
        os.makedirs('model_hands')
    
    np.save(os.path.join('model_hands', 'scaler_params.npy'), scaler_params)
    
    return X_scaled, y, scaler_params

def create_improved_model(input_shape, num_classes=26):
    model = tf.keras.Sequential([
        # camada de entrada
        tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # camadas intermediarias
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # camada de saida
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model_with_cross_validation():
    print("=== TREINAMENTO DO MODELO DE LIBRAS ===")
    print("Carregando dados de treinamento...")
    
    # carregar dados
    X, y, scaler_params = load_training_data()
    if X is None:
        return
    
    print(f"Forma dos dados: {X.shape}")
    print(f"Numero de classes: {len(np.unique(y))}")
    
    # configurar validacao cruzada
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_histories = []
    fold_accuracies = []
    
    print(f"\nIniciando validacao cruzada com {n_splits} folds...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        # dividir dados
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Dados de treinamento: {X_train.shape[0]} amostras")
        print(f"Dados de validacao: {X_val.shape[0]} amostras")
        
        # criar modelo
        model = create_improved_model((X.shape[1],))
        
        # callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # treinar modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # avaliar modelo
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_accuracy)
        fold_histories.append(history)
        
        print(f"Acuracia do fold {fold + 1}: {val_accuracy:.4f}")
    
    # resultados da validacao cruzada
    print(f"\n=== RESULTADOS DA VALIDACAO CRUZADA ===")
    print(f"Acuracia media: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Melhor acuracia: {np.max(fold_accuracies):.4f}")
    print(f"Pior acuracia: {np.min(fold_accuracies):.4f}")
    
    # treinar modelo final com todos os dados
    print(f"\n=== TREINAMENTO DO MODELO FINAL ===")
    final_model = create_improved_model((X.shape[1],))
    
    # dividir dados para treinamento final
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dados de treinamento final: {X_train.shape[0]} amostras")
    print(f"Dados de teste: {X_test.shape[0]} amostras")
    
    # callbacks para modelo final
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7
    )
    
    # treinar modelo final
    final_history = final_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # avaliar modelo final
    test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcuracia final no teste: {test_accuracy:.4f}")
    
    # fazer predicoes
    y_pred = np.argmax(final_model.predict(X_test), axis=1)
    
    # relatorio de classificacao
    print("\n=== RELATORIO DE CLASSIFICACAO ===")
    # usar apenas as classes que existem nos dados
    unique_classes = sorted(np.unique(y))
    target_names = [chr(i + ord('A')) for i in unique_classes]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    print("\n=== MATRIZ DE CONFUSAO ===")
    print("Formato: [Linha = Verdadeiro, Coluna = Predito]")
    print("Classes:", target_names)
    print(cm)
    
    # salvar modelo
    model_path = os.path.join('model_hands', 'libras_model_improved.h5')
    final_model.save(model_path)
    print(f"\nModelo salvo em: {model_path}")
    
    # plotar historico de treinamento
    plot_training_history(final_history, fold_histories)
    
    return final_model, test_accuracy

def plot_training_history(final_history, fold_histories):
    # criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Historico de Treinamento - Modelo de Libras', fontsize=16)
    
    # plot 1: acuracia do modelo final
    axes[0, 0].plot(final_history.history['accuracy'], label='Treinamento')
    axes[0, 0].plot(final_history.history['val_accuracy'], label='Validacao')
    axes[0, 0].set_title('Acuracia - Modelo Final')
    axes[0, 0].set_xlabel('Epoca')
    axes[0, 0].set_ylabel('Acuracia')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # plot 2: loss do modelo final
    axes[0, 1].plot(final_history.history['loss'], label='Treinamento')
    axes[0, 1].plot(final_history.history['val_loss'], label='Validacao')
    axes[0, 1].set_title('Loss - Modelo Final')
    axes[0, 1].set_xlabel('Epoca')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # plot 3: acuracia dos folds
    for i, history in enumerate(fold_histories):
        axes[1, 0].plot(history.history['val_accuracy'], label=f'Fold {i+1}', alpha=0.7)
    axes[1, 0].set_title('Acuracia - Validacao Cruzada')
    axes[1, 0].set_xlabel('Epoca')
    axes[1, 0].set_ylabel('Acuracia')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # plot 4: loss dos folds
    for i, history in enumerate(fold_histories):
        axes[1, 1].plot(history.history['val_loss'], label=f'Fold {i+1}', alpha=0.7)
    axes[1, 1].set_title('Loss - Validacao Cruzada')
    axes[1, 1].set_xlabel('Epoca')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # salvar figura
    plot_path = os.path.join('model_hands', 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvo em: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    train_model_with_cross_validation() 