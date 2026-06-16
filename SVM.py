"""
REVISI: Deteksi Dini Penyakit Jantung Koroner (CHD) menggunakan Support Vector Machine (SVM)
Berdasarkan Metodologi Jurnal: "Early Coronary Heart Disease Deciphered via Support Vector Machines"
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Buat folder hasil
os.makedirs('hasil', exist_ok=True)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ============================================================
# BAGIAN 1: PREPROCESSING
# ============================================================
print("=" * 60)
print("1. PREPROCESSING")
print("=" * 60)

# 1.1 Load Dataset
df = pd.read_csv('heart.csv')
print(f"[+] Dataset Dimuat: {df.shape[0]} pasien, {df.shape[1]-1} fitur input.")

# 1.2 Label Encoding (sesuai artikel)
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
le = LabelEncoder()
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
print(f"[+] Label Encoding diterapkan pada: {cat_cols}")

X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# 1.3 Normalisasi Data (StandardScaler)
scaler = StandardScaler()
X_selected = scaler.fit_transform(X)
print(f"[+] Normalisasi StandardScaler diterapkan. Jumlah fitur: {X.shape[1]}")

# ============================================================
# BAGIAN 2: KONFIGURASI MODEL SVM (parameter default sesuai artikel)
# ============================================================
kernels = {
    'Linear'    : SVC(kernel='linear',  probability=True, random_state=42),
    'Polynomial': SVC(kernel='poly',    probability=True, random_state=42, degree=3),
    'RBF'       : SVC(kernel='rbf',     probability=True, random_state=42),
    'Sigmoid'   : SVC(kernel='sigmoid', probability=True, random_state=42),
}

# ============================================================
# BAGIAN 3: UJI COBA 1 - K-FOLD CROSS VALIDATION (MULTI-K)
# ============================================================
K_VALUES = [2, 3, 4, 6, 7, 8, 9, 10]  # Semua nilai K yang diuji

all_kfold_results = {}  # { K: DataFrame hasil }

for K in K_VALUES:
    print("\n" + "=" * 60)
    print(f"2. UJI COBA 1: {K}-FOLD CROSS VALIDATION")
    print("=" * 60)

    cvK = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    results_kfold = {}

    for name, model in kernels.items():
        auc_list, acc_list, sens_list, spec_list = [], [], [], []

        for train_idx, test_idx in cvK.split(X_selected, y):
            X_tr, X_te = X_selected[train_idx], X_selected[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()

            acc_list.append(accuracy_score(y_te, y_pred))
            auc_list.append(roc_auc_score(y_te, y_prob))
            sens_list.append(tp / (tp + fn))
            spec_list.append(tn / (tn + fp))

        results_kfold[name] = {
            'AUC'        : round(np.mean(auc_list) * 100, 2),
            'Accuracy'   : round(np.mean(acc_list) * 100, 2),
            'Sensitivity': round(np.mean(sens_list) * 100, 2),
            'Specificity': round(np.mean(spec_list) * 100, 2)
        }

    df_kfold = pd.DataFrame(results_kfold).T
    df_kfold.index.name = 'Model'
    all_kfold_results[K] = df_kfold
    print(df_kfold.to_string())

# Ringkasan semua K (semua kernel)
summary_rows = []
for K, df_k in all_kfold_results.items():
    for kernel_name in df_k.index:
        row = df_k.loc[kernel_name].copy()
        row['K'] = K
        row['Kernel'] = kernel_name
        summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)[['K', 'Kernel', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity']]

# Tabel rata-rata Accuracy per K (dasar pemilihan K terbaik)
print("\n" + "=" * 60)
print("PEMILIHAN K TERBAIK — Rata-rata Accuracy semua kernel per K")
print("=" * 60)
print(f"{'K-Fold':<10} {'Rata-rata Accuracy':>20}  {'Kernel Terbaik':>15}  {'Accuracy':>10}")
print("-" * 60)
for k in K_VALUES:
    df_k = all_kfold_results[k]
    mean_acc = df_k['Accuracy'].mean()
    best_kernel = df_k['Accuracy'].idxmax()
    best_acc = df_k.loc[best_kernel, 'Accuracy']
    print(f"{k}-Fold    {mean_acc:>18.2f}%  {best_kernel:>15}  {best_acc:>9.2f}%")

best_K = max(K_VALUES, key=lambda k: all_kfold_results[k]['Accuracy'].mean())
print(f"\n  --> K terbaik dipilih: {best_K}-Fold (rata-rata Accuracy tertinggi = {all_kfold_results[best_K]['Accuracy'].mean():.2f}%)")

# ============================================================
# BAGIAN 4: UJI COBA 2 - LEAVE ONE OUT (LOO) CROSS VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("3. UJI COBA 2: LEAVE ONE OUT CROSS VALIDATION (TABEL 3)")
print("=" * 60)
print("Memproses LOO CV... (Mohon tunggu, ini membutuhkan waktu)")

loo = LeaveOneOut()
results_loo = {}

for name, model in kernels.items():
    y_true_all, y_pred_all, y_prob_all = [], [], []
    
    for train_idx, test_idx in loo.split(X_selected):
        X_tr, X_te = X_selected[train_idx], X_selected[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_tr, y_tr)
        y_pred_all.extend(model.predict(X_te))
        y_prob_all.extend(model.predict_proba(X_te)[:, 1])
        y_true_all.extend(y_te)

    # Kalkulasi metrik global untuk LOO
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    
    acc = accuracy_score(y_true_all, y_pred_all) * 100
    auc = roc_auc_score(y_true_all, y_prob_all) * 100
    sens = (tp / (tp + fn)) * 100
    spec = (tn / (tn + fp)) * 100

    results_loo[name] = {
        'AUC'        : round(auc, 2),
        'Accuracy'   : round(acc, 2),
        'Sensitivity': round(sens, 2),
        'Specificity': round(spec, 2)
    }

df_loo = pd.DataFrame(results_loo).T
df_loo.index.name = 'Model'
print("\n" + df_loo.to_string())

# ============================================================
# VISUALISASI - GRAFIK PER KERNEL ACROSS K VALUES
# ============================================================
metrics   = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
colors    = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
k_labels  = [f"{k}-Fold" for k in K_VALUES]
kernel_names = list(kernels.keys())

# ============================================================
# VISUALISASI - SATU GRAFIK RINGKAS (2 PANEL)
# ============================================================

# Cari K terbaik berdasarkan rata-rata Accuracy semua kernel
best_K = max(K_VALUES, key=lambda k: all_kfold_results[k]['Accuracy'].mean())
df_best_kfold = all_kfold_results[best_K]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Perbandingan Performa SVM — Semua Kernel', fontsize=14, fontweight='bold')

metric_colors = {'AUC': '#2196F3', 'Accuracy': '#4CAF50', 'Sensitivity': '#FF9800', 'Specificity': '#E91E63'}
x     = np.arange(len(kernel_names))
width = 0.2

# Panel kiri: K-Fold terbaik
ax1 = axes[0]
for i, metric in enumerate(metrics):
    vals = [df_best_kfold.loc[k, metric] for k in kernel_names]
    bars = ax1.bar(x + i * width, vals, width, label=metric, color=list(metric_colors.values())[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7)
ax1.set_title(f'K-Fold CV Terbaik ({best_K}-Fold)', fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(kernel_names)
ax1.set_ylabel('Persentase (%)')
ax1.set_ylim(70, 100)
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# Panel kanan: LOO
ax2 = axes[1]
for i, metric in enumerate(metrics):
    vals = [df_loo.loc[k, metric] for k in kernel_names]
    bars = ax2.bar(x + i * width, vals, width, label=metric, color=list(metric_colors.values())[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=7)
ax2.set_title('Leave One Out (LOO) CV', fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(kernel_names)
ax2.set_ylabel('Persentase (%)')
ax2.set_ylim(70, 100)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('hasil/performa_svm.png', dpi=150, bbox_inches='tight')
print("  --> Disimpan: hasil/performa_svm.png")
plt.close()

# ============================================================
# KESIMPULAN
# ============================================================
print("\n" + "=" * 60)
best_kfold_kernel = df_best_kfold['Accuracy'].idxmax()
best_loo_kernel   = df_loo['Accuracy'].idxmax()
print(f"Berdasarkan eksperimen:")
print(f"- K-Fold terbaik : {best_K}-Fold")
print(f"- Kernel terbaik ({best_K}-Fold) : {best_kfold_kernel} — Accuracy {df_best_kfold.loc[best_kfold_kernel, 'Accuracy']}%")
print(f"- Kernel terbaik (LOO)    : {best_loo_kernel} — Accuracy {df_loo.loc[best_loo_kernel, 'Accuracy']}%")
print("\nFile tersimpan di folder 'hasil/':")
print("  PNG  : performa_svm.png")
print("=" * 60)