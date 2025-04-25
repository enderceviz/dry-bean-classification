# Kütüphanelerin Yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
warnings.filterwarnings('ignore')

# Veri Yükleme
df = pd.read_excel("data/Dry_Bean_Dataset.xlsx")

# Eksik Veri Ekleme
for col in ['Area', 'Perimeter']:
    df.loc[df.sample(frac=0.05, random_state=42).index, col] = np.nan
df.loc[df.sample(frac=0.35, random_state=42).index, 'MajorAxisLength'] = np.nan

# Eksik Verilerin Kontrolü
print("Eksik Veri Kontrolü:")
print(df.isnull().sum())

# %5 Eksik Verileri Ortalama ile Doldurma
df['Area'] = df['Area'].fillna(df['Area'].mean())
df['Perimeter'] = df['Perimeter'].fillna(df['Perimeter'].mean())

# %35 Eksik Verileri Satır Bazlı Silme
df = df.dropna(subset=['MajorAxisLength'])

# Aykırı Değerlerin Temizlenmesi (IQR Yöntemi)
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Tüm sayısal sütunlar için aykırı değer temizleme
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df = remove_outliers(df, numeric_columns)

# Özellik Ölçekleme ve Label Encoding
X = df.drop(columns=["Class"])
y = df["Class"]

# StandardScaler kullanarak ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# PCA Uygulaması ve Scatter Plot
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
threshold = pca.explained_variance_ratio_.mean()
num_components = sum(pca.explained_variance_ratio_ > threshold)
pca_final = pd.DataFrame(X_pca[:, :num_components], columns=[f'PC{i+1}' for i in range(num_components)])
pca_final['Class'] = y_encoded

# PCA Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_final["PC1"], pca_final["PC2"], c=y_encoded, cmap='tab10')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: İlk İki Bileşen Dağılımı")
plt.grid(True)
plt.savefig("results/pca_plot.png")
plt.show()

# LDA Uygulaması ve Scatter Plot
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y_encoded)
lda_final = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(X_lda.shape[1])])
lda_final['Class'] = y_encoded

# LDA Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(lda_final["LD1"], lda_final["LD2"], c=y_encoded, cmap='tab10')
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA: İlk İki Bileşen Dağılımı")
plt.grid(True)
plt.savefig("results/lda_plot.png")
plt.show()

# Model Parametreleri
param_grids = {
    "Logistic Regression": {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']
    },
    "Decision Tree": {
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    "Naive Bayes": {}
}

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
    "Naive Bayes": GaussianNB()
}

# Nested Cross-Validation ile Modelleme
def nested_cv(X, y, model, param_grid, outer_splits=5, inner_splits=3):
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    confusion_matrices = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if param_grid:
            clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy')
        else:
            clf = model

        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_ if isinstance(clf, GridSearchCV) else clf
        y_pred = best_model.predict(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)
            scores['roc_auc'].append(roc_auc_score(y_test, y_proba, multi_class='ovr'))

    # Confusion Matrix Görselleştirme
    avg_cm = np.mean(confusion_matrices, axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Ortalama Confusion Matrix')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    return {k: (np.mean(v), np.std(v)) for k, v in scores.items()}

# Her veri temsili için modelleme
data_representations = {
    "Ham Veri": X_scaled,
    "PCA": X_pca,
    "LDA": X_lda
}

all_results = {}

for data_name, X_data in data_representations.items():
    print(f"\n{data_name} için modelleme başlıyor...")
    results = {}
    for model_name, model in models.items():
        print(f"Model: {model_name} işleniyor...")
        results[model_name] = nested_cv(X_data, y_encoded, model, param_grids[model_name])
    all_results[data_name] = results

# Performans Sonuçlarını DataFrame'e çevirip CSV'ye kaydet
performance_data = []
for data_name, models_results in all_results.items():
    for model_name, metrics in models_results.items():
        performance_data.append({
            "Veri Temsili": data_name,
            "Model": model_name,
            "Mean Accuracy": metrics['accuracy'][0],
            "Std Accuracy": metrics['accuracy'][1],
            "Mean Precision": metrics['precision'][0],
            "Std Precision": metrics['precision'][1],
            "Mean Recall": metrics['recall'][0],
            "Std Recall": metrics['recall'][1],
            "Mean F1": metrics['f1'][0],
            "Std F1": metrics['f1'][1],
            "Mean ROC-AUC": metrics['roc_auc'][0] if 'roc_auc' in metrics else np.nan,
            "Std ROC-AUC": metrics['roc_auc'][1] if 'roc_auc' in metrics else np.nan,
        })

performance_table = pd.DataFrame(performance_data)
performance_table.to_csv("results/performance_table.csv", index=False)
print("\nPerformans Sonuçları:")
print(performance_table)

# ROC Curve Çizimi (OVA Yöntemiyle)
for data_name, X_data in data_representations.items():
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
    model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    n_classes = len(np.unique(y_encoded))
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = plt.colormaps.get_cmap('tab10')

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})", color=colors(i / n_classes))

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Eğrileri ({data_name} - XGBoost - One-vs-All)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"results/roc_curve_{data_name}.png")
    plt.show()