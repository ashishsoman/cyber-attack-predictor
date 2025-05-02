import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, precision_recall_curve

import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras import models, layers

from utils import weighted_bce, months, industries, countries, industry_risk, country_risk, recent_attacks_dict, geopolitical_tension_dict

# ----------------------------
# 1. Generate Synthetic Data
# ----------------------------

np.random.seed(42) 


data = []

for month in range(1, len(months) + 1):
    for industry in industries:
        for country in countries:
            cybersecurity_posture = np.random.uniform(0, 1)
            threat_intelligence = np.random.uniform(0, 1)
            sector_vulnerability_index = np.random.uniform(0,1)
            is_high_risk_country = 1 if country in country_risk else 0
            industry_score = industry_risk.get(industry, 0.5)
            # Base risk influenced by country and industry
            base_risk = 0.2 + 0.1 * is_high_risk_country  + 0.1 * industry_score - 0.1 * cybersecurity_posture - 0.1 * threat_intelligence

            recent_attacks = recent_attacks_dict.get(industry, 2) # Randomized lag feature
            geopolitical_tension = geopolitical_tension_dict.get(country, 2)  # External feature
            season = (month % 12) / 12.0  # Encoded seasonal feature
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            # cybersecurity of particular industry for a specific country
            

            # Target: 1 if risk + noise > threshold
            risk_score = base_risk + 0.05 * recent_attacks + 0.1 * geopolitical_tension + np.random.normal(0, 0.1)
            label = int(risk_score > 0.5)

            data.append({
                'month':month,
                'month_cos':month_cos,
                'industry': industry,
                'country': country,
                'recent_attacks': recent_attacks,
                'geopolitical_tension': geopolitical_tension,
                'cybersecurity_posture': cybersecurity_posture,
                'season': season,
                'label': label
            })

df = pd.DataFrame(data)

print(recent_attacks)

# ----------------------------
# 2. Preprocessing
# ----------------------------

le_industry = LabelEncoder()
le_country = LabelEncoder()

df['industry_enc'] = le_industry.fit_transform(df['industry'])
df['country_enc'] = le_country.fit_transform(df['country'])

features = ['industry_enc', 'country_enc','recent_attacks','geopolitical_tension','cybersecurity_posture']
X = df[features].values
y = df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ----------------------------
# 3. Model Definition
# ----------------------------

model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=weighted_bce(pos_weight=2.0),
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

class_weight = {0: 1.0, 1: 
                3.0}

# ----------------------------
# 4. Train the Model
# ----------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=70,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
    class_weight=class_weight,
)

# ----------------------------
# 5. Evaluation
# ----------------------------

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.4).astype(int)

# Try multiple thresholds
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_thresh = 0.3



print("Threshold Tuning Results:")
for thresh in thresholds:
    y_pred_thresh = (y_pred_prob > thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    print(f"Threshold: {thresh:.2f} | F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

prec, rec, thresh = precision_recall_curve(y_test, y_pred_prob)
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
best_thresh = thresh[np.argmax(f1)]

print(f"\n✅ Best Threshold = {best_thresh:.2f} with F1 Score = {best_f1:.4f}")

y_final_pred = (y_pred_prob > best_thresh).astype(int)
cm = confusion_matrix(y_test, y_final_pred)

print(cm)

'''
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Attack', 'Attack'], yticklabels=['No Attack', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold = {best_thresh:.2f})')
plt.show()
 '''
print("\nClassification Report:")
print(classification_report(y_test, y_final_pred))

print("F1 Score:", f1_score(y_test, y_final_pred))

# Single input sample with 4 features
# 'industry_enc', 'country_enc','recent_attacks','geopolitical_tension','cybersecurity_posture'
X_input = np.array([['0', '0','5','0.8','0.2']], dtype=np.float32)  # shape = (1, 4)

print(model.predict(X_input))

# After training
model.save("model.keras")  # HDF5 format



