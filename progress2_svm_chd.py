"""
Progress 2: Metodologi Penelitian + Uji Coba
Deteksi Dini Penyakit Jantung Koroner (CHD) menggunakan Support Vector Machine (SVM)
Dataset: heart_rapi.xlsx (918 pasien, 12 atribut)
Referensi: Akhtar et al., 2023 IEEE ICCWAMTIP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (cross_val_score, LeaveOneOut, StratifiedKFold,
                                     train_test_split)
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# ============================================================
# BAGIAN 1: PREPROCESSING DATA
# ============================================================
print("=" * 60)
print("METODOLOGI PENELITIAN")
print("=" * 60)

# 1.1 Load Dataset
df = pd.read_excel('heart_rapi.xlsx')
print(f"\n[1] Dataset Dimuat")
print(f"    Jumlah data  : {df.shape[0]} pasien")
print(f"    Jumlah fitur : {df.shape[1] - 1} atribut + 1 target")
print(f"    Distribusi target: {df['HeartDisease'].value_counts().to_dict()}")
print(f"    (0 = Tidak CHD, 1 = CHD)")

# 1.2 Encoding variabel kategorikal
print("\n[2] Encoding Variabel Kategorikal")
le = LabelEncoder()
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_enc = df.copy()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col])
    print(f"    {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 1.3 Pisahkan fitur dan target
X = df_enc.drop('HeartDisease', axis=1)
y = df_enc['HeartDisease']
print(f"\n[3] Fitur Input (X): {list(X.columns)}")
print(f"    Target (y): HeartDisease")

# 1.4 Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n[4] Normalisasi: StandardScaler diterapkan")

# ============================================================
# BAGIAN 2: MODEL SVM
# ============================================================
print("\n" + "=" * 60)
print("MODEL SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

kernels = {
    'Linear'    : SVC(kernel='linear',  probability=True, random_state=42),
    'Polynomial': SVC(kernel='poly',    probability=True, random_state=42, degree=3),
    'RBF'       : SVC(kernel='rbf',     probability=True, random_state=42),
    'Sigmoid'   : SVC(kernel='sigmoid', probability=True, random_state=42),
}

# ============================================================
# BAGIAN 3: UJI COBA - TRAIN-TEST SPLIT (80/20)
# ============================================================
print("\n" + "=" * 60)
print("UJI COBA 1: TRAIN-TEST SPLIT (80% Latih / 20% Uji)")
print("=" * 60)

# Bagi data: 80% untuk training, 20% untuk testing.
# stratify=y menjaga proporsi kelas (CHD / Tidak CHD) tetap sama di kedua bagian.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Data latih : {X_train.shape[0]} pasien (80%)")
print(f"  Data uji   : {X_test.shape[0]} pasien (20%)")

results_tts = {}
for name, model in kernels.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_prob) * 100
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = (tp / (tp + fn)) * 100
    spec = (tn / (tn + fp)) * 100

    results_tts[name] = {
        'AUC'        : round(auc, 2),
        'Accuracy'   : round(acc, 2),
        'Sensitivity': round(sens, 2),
        'Specificity': round(spec, 2),
    }
    print(f"\n  [{name}]")
    print(f"    AUC         : {results_tts[name]['AUC']:.2f}%")
    print(f"    Accuracy    : {results_tts[name]['Accuracy']:.2f}%")
    print(f"    Sensitivity : {results_tts[name]['Sensitivity']:.2f}%")
    print(f"    Specificity : {results_tts[name]['Specificity']:.2f}%")

df_tts = pd.DataFrame(results_tts).T
df_tts.index.name = 'Model'
print("\nTabel 1. Performa CHD menggunakan SVM Train-Test Split (80/20):")
print(df_tts.to_string())

# ============================================================
# BAGIAN 4: UJI COBA - 5-FOLD CROSS VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("UJI COBA 2: 5-FOLD CROSS VALIDATION")
print("=" * 60)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_5fold = {}

for name, model in kernels.items():
    acc_scores  = cross_val_score(model, X_scaled, y, cv=cv5, scoring='accuracy')
    auc_scores  = cross_val_score(model, X_scaled, y, cv=cv5, scoring='roc_auc')
    sens_scores = cross_val_score(model, X_scaled, y, cv=cv5, scoring='recall')  # sensitivity = recall

    # Specificity per fold
    spec_list = []
    for train_idx, test_idx in cv5.split(X_scaled, y):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
        spec_list.append(tn / (tn + fp))

    results_5fold[name] = {
        'AUC'        : round(auc_scores.mean() * 100, 2),
        'Accuracy'   : round(acc_scores.mean() * 100, 2),
        'Sensitivity': round(sens_scores.mean() * 100, 2),
        'Specificity': round(np.mean(spec_list) * 100, 2),
    }
    print(f"\n  [{name}]")
    print(f"    AUC         : {results_5fold[name]['AUC']:.2f}%")
    print(f"    Accuracy    : {results_5fold[name]['Accuracy']:.2f}%")
    print(f"    Sensitivity : {results_5fold[name]['Sensitivity']:.2f}%")
    print(f"    Specificity : {results_5fold[name]['Specificity']:.2f}%")

# Tabel hasil 5-fold
df_5fold = pd.DataFrame(results_5fold).T
df_5fold.index.name = 'Model'
print("\nTabel 2. Performa CHD menggunakan SVM 5-Fold:")
print(df_5fold.to_string())

# ============================================================
# BAGIAN 5: UJI COBA - LEAVE ONE OUT CROSS VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("UJI COBA 3: LEAVE ONE OUT CROSS VALIDATION")
print("=" * 60)
print("(Proses LOO memerlukan waktu lebih lama...)")

loo = LeaveOneOut()
results_loo = {}

for name, model in kernels.items():
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    acc  = accuracy_score(y_true_all, y_pred_all) * 100
    auc  = roc_auc_score(y_true_all, y_prob_all) * 100
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    sens = (tp / (tp + fn)) * 100
    spec = (tn / (tn + fp)) * 100

    results_loo[name] = {
        'AUC'        : round(auc, 2),
        'Accuracy'   : round(acc, 2),
        'Sensitivity': round(sens, 2),
        'Specificity': round(spec, 2),
    }
    print(f"\n  [{name}]")
    print(f"    AUC         : {results_loo[name]['AUC']:.2f}%")
    print(f"    Accuracy    : {results_loo[name]['Accuracy']:.2f}%")
    print(f"    Sensitivity : {results_loo[name]['Sensitivity']:.2f}%")
    print(f"    Specificity : {results_loo[name]['Specificity']:.2f}%")

df_loo = pd.DataFrame(results_loo).T
df_loo.index.name = 'Model'
print("\nTabel 3. Performa CHD menggunakan SVM Leave One Out:")
print(df_loo.to_string())

# ============================================================
# BAGIAN 6: VISUALISASI
# ============================================================
print("\n" + "=" * 60)
print("VISUALISASI HASIL")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Performa SVM untuk Deteksi CHD', fontsize=14, fontweight='bold')

metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
x = np.arange(len(metrics))
width = 0.2
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

# Plot Train-Test Split
ax0 = axes[0]
for i, (name, vals) in enumerate(results_tts.items()):
    vals_list = [vals[m] for m in metrics]
    bars = ax0.bar(x + i * width, vals_list, width, label=name, color=colors[i], alpha=0.85)
    for bar, v in zip(bars, vals_list):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7)
ax0.set_title('Train-Test Split (80/20)')
ax0.set_xticks(x + width * 1.5)
ax0.set_xticklabels(metrics)
ax0.set_ylabel('Persentase (%)')
ax0.set_ylim(70, 105)
ax0.legend(fontsize=8)
ax0.grid(axis='y', alpha=0.3)

# Plot 5-Fold
ax1 = axes[1]
for i, (name, vals) in enumerate(results_5fold.items()):
    vals_list = [vals[m] for m in metrics]
    bars = ax1.bar(x + i * width, vals_list, width, label=name, color=colors[i], alpha=0.85)
    for bar, v in zip(bars, vals_list):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7)
ax1.set_title('5-Fold Cross Validation')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(metrics)
ax1.set_ylabel('Persentase (%)')
ax1.set_ylim(70, 105)
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# Plot LOO
ax2 = axes[2]
for i, (name, vals) in enumerate(results_loo.items()):
    vals_list = [vals[m] for m in metrics]
    bars = ax2.bar(x + i * width, vals_list, width, label=name, color=colors[i], alpha=0.85)
    for bar, v in zip(bars, vals_list):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7)
ax2.set_title('Leave One Out Cross Validation')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(metrics)
ax2.set_ylabel('Persentase (%)')
ax2.set_ylim(70, 105)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('hasil_performa_svm.png', dpi=150, bbox_inches='tight')
print("  Grafik performa disimpan: hasil_performa_svm.png")

# Heatmap korelasi fitur
plt.figure(figsize=(10, 8))
corr = df_enc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Heatmap Korelasi Fitur Dataset CHD', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_korelasi.png', dpi=150, bbox_inches='tight')
print("  Heatmap korelasi disimpan: heatmap_korelasi.png")

# Confusion Matrix untuk model terbaik (Sigmoid LOO)
best_model = SVC(kernel='sigmoid', probability=True, random_state=42)
best_model.fit(X_scaled, y)

y_pred_final = best_model.predict(X_scaled)
cm = confusion_matrix(y, y_pred_final)
plt.figure(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak CHD', 'CHD'])
disp.plot(colorbar=False, cmap='Blues')
plt.title('Confusion Matrix - SVM Sigmoid (Terbaik)', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_sigmoid.png', dpi=150, bbox_inches='tight')
print("  Confusion matrix disimpan: confusion_matrix_sigmoid.png")

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print("\n" + "=" * 60)
print("RINGKASAN HASIL")
print("=" * 60)
print("\nTabel 1 - Train-Test Split (80/20):")
print(df_tts.to_string())
print("\nTabel 2 - 5-Fold Cross Validation:")
print(df_5fold.to_string())
print("\nTabel 3 - Leave One Out CV:")
print(df_loo.to_string())

best_tts   = df_tts['Accuracy'].idxmax()
best_5fold = df_5fold['Accuracy'].idxmax()
best_loo   = df_loo['Accuracy'].idxmax()
print(f"\nModel terbaik (Train-Test)   : {best_tts} - Accuracy {df_tts.loc[best_tts,'Accuracy']:.2f}%")
print(f"Model terbaik (5-Fold)       : {best_5fold} - Accuracy {df_5fold.loc[best_5fold,'Accuracy']:.2f}%")
print(f"Model terbaik (Leave One Out): {best_loo}   - Accuracy {df_loo.loc[best_loo,'Accuracy']:.2f}%")
print("\nSelesai! Semua file output telah disimpan.")
plt.show()
