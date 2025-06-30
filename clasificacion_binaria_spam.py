import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
# Para el preprocesamiento de texto
from sklearn.feature_extraction.text import TfidfVectorizer
import re  # Para expresiones regulares
import string  # Para manipulación de cadenas

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. Cargar el Dataset de Spam ---
# Asegúrate de que 'spam.csv' esté en el mismo directorio o proporciona la ruta completa.
# Este dataset suele tener dos columnas principales: 'v1' (etiqueta) y 'v2' (mensaje)
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')  # 'latin-1' suele ser la codificación para este archivo
    # Renombrar columnas para mayor claridad
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    # Seleccionar solo las columnas relevantes y eliminar las extras que suelen venir vacías
    df = df[['label', 'message']]
    print("Dataset 'spam.csv' cargado exitosamente.")
    print("Primeras 5 filas del dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'spam.csv' no encontrado. Asegúrate de que el archivo esté en la ruta correcta.")
    print("Puedes descargarlo de Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    exit()
except Exception as e:
    print(f"Error al cargar o procesar el CSV: {e}")
    exit()

# --- 2. Preparar las etiquetas (convertir 'ham'/'spam' a 0/1) ---
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
y = df['label']  # Las etiquetas numéricas

print("\n--- Balance de Clases en el Dataset Original ---")
total_samples = len(y)
num_ham = y.value_counts().get(0, 0)
num_spam = y.value_counts().get(1, 0)

print(f"Clase 0 (Ham/No Spam): {num_ham} muestras ({num_ham / total_samples * 100:.4f}%)")
print(f"Clase 1 (Spam): {num_spam} muestras ({num_spam / total_samples * 100:.4f}%)")
if num_spam > 0:
    print(f"Relación Ham / Spam: {num_ham / num_spam:.2f} veces más mensajes Ham.")
else:
    print("¡Advertencia: No hay mensajes Spam en el dataset! Esto es inusual.")
print("--- Fin: Análisis de Balance de Clases ---")


# --- 3. Preprocesamiento de Texto y Vectorización TF-IDF ---

def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Eliminar puntuación
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = text.strip()  # Eliminar espacios en blanco al inicio/final
    return text


# Aplicar la limpieza a los mensajes
df['cleaned_message'] = df['message'].apply(clean_text)

print("\nEjemplo de mensaje original:", df['message'].iloc[0])
print("Ejemplo de mensaje limpio:", df['cleaned_message'].iloc[0])

# Inicializar TfidfVectorizer
# max_features: Limita el número de características (palabras) para evitar un vector demasiado grande.
#               Esto es importante para el tamaño de entrada de tu red neuronal.
# stop_words: Elimina palabras comunes que no aportan mucho significado (como 'el', 'la', 'y').
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Ajustar y transformar los mensajes limpios en vectores TF-IDF
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_message'])

# Convertir la matriz dispersa TF-IDF a un array denso de NumPy
X = X_tfidf.toarray()

print(f"\nDimensiones de X (muestras, características TF-IDF): {X.shape}")
print(f"Número de palabras (características) generadas por TF-IDF: {X.shape[1]}")
print("Primeras 5 filas del array TF-IDF (ejemplo, puede ser disperso si no hay mucha actividad):")
print(X[:5])

# --- 4. Separar train, val y test (usando X como el array TF-IDF y y como las etiquetas) ---
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42,
                                                  stratify=y_train_full)

print("\n--- Dimensiones de los conjuntos después de la división ---")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- Verificación de Balance de Clases en los Subconjuntos ---
print("\n--- Balance de Clases en el Conjunto de Entrenamiento (y_train) ---")
print(y_train.value_counts(normalize=True).mul(100).apply(lambda x: f"{x:.4f}%"))
print("\n--- Balance de Clases en el Conjunto de Validación (y_val) ---")
print(y_val.value_counts(normalize=True).mul(100).apply(lambda x: f"{x:.4f}%"))
print("\n--- Balance de Clases en el Conjunto de Test (y_test) ---")
print(y_test.value_counts(normalize=True).mul(100).apply(lambda x: f"{x:.4f}%"))
print("--- Fin: Verificación de Balance de Clases en los Subconjuntos ---")

# --- A partir de aquí, el código es muy similar al de fraude de tarjetas de crédito ---

# --- 5. Convertir a tensores de PyTorch ---
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# --- 6. Crear DataLoaders para manejar los datos en lotes ---
# El tamaño del lote podría ser más pequeño dependiendo de la memoria RAM disponible
# dado que TF-IDF genera vectores dispersos que se convierten a densos.
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)  # Reducido batch_size
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512)

# --- 7. Definir el Modelo de Red Neuronal ---
# El input_size será el número de características generadas por TF-IDF
input_size_nn = X.shape[1]


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)  # Tasa de Dropout ajustada de la mejora anterior
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = Net(input_size=input_size_nn)  # Usar el input_size determinado por TF-IDF

# --- 8. Entrenamiento del Modelo con Early Stopping y Ponderación de Clases ---
# Cálculo de pos_weight para balancear la pérdida (peso para la clase positiva/minoritaria)
# count(clase_0) / count(clase_1)
pos_weight_multiplier = 1.2  # Multiplicador de la mejora anterior
pos_weight = torch.tensor([num_ham / num_spam * pos_weight_multiplier], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # LR y L2 Reg. de la mejora anterior

epochs = 50
train_losses, val_losses = [], []

print("\n\n--- Arquitectura del Modelo ---")
print(model)

# Parámetros de Early Stopping
patience = 15
min_delta = 0.00001
best_val_loss = float('inf')
epochs_no_improve = 0
model_save_path = "best_spam_detection_model.pth"  # Nuevo nombre para el modelo
best_model_state = None

print("\n--- Comienza el Entrenamiento de la Detección de Spam ---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred_logits = model(X_batch)
        loss = criterion(y_pred_logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred_logits = model(X_batch)
            loss = criterion(y_pred_logits, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
        torch.save(best_model_state, model_save_path)
        print(f"  --> Mejor modelo guardado con Val Loss: {best_val_loss:.6f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping activado en la epoch {epoch + 1}. No hubo mejora en {patience} epochs.")
            break

if best_model_state:
    model.load_state_dict(best_model_state)
    print("\nModelo restaurado al mejor estado guardado durante el entrenamiento para la evaluación final.")
else:
    print("\nNo se encontró ningún mejor estado de modelo para cargar. Se utilizará el estado final del modelo.")

# --- 9. Evaluación Final en el Conjunto de Test ---
print("\n--- Evaluando el modelo con el conjunto de test (datos nunca vistos) ---")
model.eval()
y_true_test_np = []
y_pred_test_np = []
y_pred_probs_test_np = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred_logits = model(X_batch)
        y_pred_probs = torch.sigmoid(y_pred_logits)
        y_predicted_labels = (y_pred_probs >= 0.5).float()

        y_true_test_np.extend(y_batch.cpu().numpy().flatten())
        y_pred_test_np.extend(y_predicted_labels.cpu().numpy().flatten())
        y_pred_probs_test_np.extend(y_pred_probs.cpu().numpy().flatten())

y_true_test_np = np.array(y_true_test_np)
y_pred_test_np = np.array(y_pred_test_np)
y_pred_probs_test_np = np.array(y_pred_probs_test_np)

accuracy = accuracy_score(y_true_test_np, y_pred_test_np)
# Precision, Recall, F1-Score para la clase minoritaria (Spam, que es 1)
precision = precision_score(y_true_test_np, y_pred_test_np, pos_label=1)
recall = recall_score(y_true_test_np, y_pred_test_np, pos_label=1)
f1 = f1_score(y_true_test_np, y_pred_test_np, pos_label=1)
conf_matrix = confusion_matrix(y_true_test_np, y_pred_test_np)

print(f"\n--- Resultados de la Evaluación en el conjunto de Test (Clase Positiva: Spam) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Spam): {precision:.4f}")
print(f"Recall (Spam): {recall:.4f}")
print(f"F1-Score (Spam): {f1:.4f}")

print("\n--- Matriz de Confusión ---")
conf_matrix_labels = ['Ham', 'Spam']
print(pd.DataFrame(conf_matrix, index=conf_matrix_labels, columns=conf_matrix_labels))

print("\n--- Reporte de Clasificación Detallado por Clase ---")
class_names = ['Ham (0)', 'Spam (1)']
print(classification_report(y_true_test_np, y_pred_test_np, target_names=class_names))

# --- 10. Visualizaciones ---
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Pérdida de Entrenamiento")
plt.plot(val_losses, label="Pérdida de Validación", linestyle='--')
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Curvas de Pérdida (Entrenamiento y Validación)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicción Ham (0)', 'Predicción Spam (1)'],
            yticklabels=['Real Ham (0)', 'Real Spam (1)'])
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.title('Matriz de Confusión')
plt.show()

precision_curve, recall_curve, _ = precision_recall_curve(y_true_test_np, y_pred_probs_test_np)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, label=f'Curva P-R (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall para Detección de Spam')
plt.legend()
plt.grid(True)
plt.show()

# --- Ejemplo de carga y predicción con el modelo guardado ---
print(f"\n--- Intentando cargar el modelo desde '{model_save_path}' para una nueva predicción ---")
try:
    loaded_model = Net(input_size=input_size_nn)  # Usar el input_size correcto
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.eval()

    print("Modelo cargado exitosamente.")

    # Buscamos una muestra de spam para ejemplificar si existe
    spam_indices = np.where(y_test.cpu().numpy().flatten() == 1)[0]
    sample_index_for_prediction = spam_indices[0] if len(spam_indices) > 0 else 0

    # Aseguramos que sample_input sea un tensor PyTorch con la forma correcta (batch_size, num_features)
    sample_input = X_test[sample_index_for_prediction].unsqueeze(0)

    with torch.no_grad():
        loaded_prediction_logits = loaded_model(sample_input)
        loaded_prediction_prob_tensor = torch.sigmoid(loaded_prediction_logits)

        loaded_prediction_prob_item = loaded_prediction_prob_tensor.item()
        loaded_prediction_label = (loaded_prediction_prob_tensor >= 0.5).float().item()

    print(f"\nPredicción con modelo cargado para la muestra de test con índice {sample_index_for_prediction}:")
    print(f"  Probabilidad de ser Spam: {loaded_prediction_prob_item:.4f}")
    print(f"  Etiqueta predicha (0=Ham, 1=Spam): {int(loaded_prediction_label)}")
    print(f"  Etiqueta real de la muestra: {int(y_test[sample_index_for_prediction].item())}")

    # Ejemplo de predicción para un nuevo mensaje de texto
    print("\n--- Ejemplo de predicción para un NUEVO mensaje de texto ---")
    new_messages = [
        "Free entry in a contest to win cash! text WIN to 80080 now!",  # Spam
        "Hey, how are you doing today? Just checking in.",  # Ham
        "URGENT! Your mobile number has won a £2000 prize. Call 09061701461.",  # Spam
        "Meeting at 2pm. Don't be late."  # Ham
    ]

    # 1. Limpiar los nuevos mensajes
    cleaned_new_messages = [clean_text(msg) for msg in new_messages]

    # 2. Transformar los mensajes limpios usando el vectorizador TF-IDF *ya ajustado*
    # NO USAR fit_transform, solo transform()
    new_messages_tfidf = tfidf_vectorizer.transform(cleaned_new_messages).toarray()

    # 3. Convertir a tensor PyTorch
    new_messages_tensor = torch.tensor(new_messages_tfidf, dtype=torch.float32)

    # 4. Realizar la predicción
    with torch.no_grad():
        new_logits = loaded_model(new_messages_tensor)
        new_probs = torch.sigmoid(new_logits)
        new_labels = (new_probs >= 0.5).float()

    print("\nResultados de predicción para nuevos mensajes:")
    for i, msg in enumerate(new_messages):
        predicted_label_text = "Spam" if new_labels[i].item() == 1 else "Ham"
        print(f"Mensaje: '{msg}'")
        print(f"  Probabilidad de Spam: {new_probs[i].item():.4f}")
        print(f"  Predicción: {predicted_label_text}\n")

except FileNotFoundError:
    print(f"¡Error: El archivo '{model_save_path}' no se encontró!")
except Exception as e:
    print(f"Error al cargar o predecir con el modelo: {e}")