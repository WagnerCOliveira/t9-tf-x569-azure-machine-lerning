# Pipeline de Machine Learning

*Última atualização: Maio 2026*

---

## Visão Geral

O pipeline de ML deste projeto cobre todo o ciclo de vida de um modelo de classificação: da ingestão de dados brutos ao registro do modelo em produção. Cada etapa é executada nos notebooks Jupyter com rastreamento automático via MLflow.

![Pipeline ML](pipeline_ml.png)

---

## Etapa 1 — Coleta de Dados

### Fontes

| Dataset | Formato | Origem |
|---------|---------|--------|
| `heart.csv` | CSV | Blob Storage Azure (`workspaceblobstore`) ou local |
| `heart_disease_uci.csv` | CSV | Local (`data/`) |

### Carregamento com Fallback

```python
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

try:
    # Tenta carregar do Blob Storage Azure
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    path = "azureml://datastores/workspaceblobstore/paths/heart.csv"
    df = pd.read_csv(path)
    print("Dataset carregado do Azure Blob Storage")
except Exception:
    # Fallback para arquivo local
    df = pd.read_csv("../data/heart.csv")
    print("Fallback: dataset carregado localmente")
```

---

## Etapa 2 — Análise Exploratória (EDA)

Realizada no notebook `data processed.ipynb`.

### Verificações Realizadas

```python
# Dimensões e tipos
df.shape          # (920, 16)
df.dtypes
df.info()

# Valores ausentes
df.isnull().sum()
df.isnull().mean()  # proporção

# Distribuição do target
df['num'].value_counts(normalize=True)

# Estatísticas descritivas
df.describe()

# Correlações
sns.heatmap(df.corr(), annot=True, fmt='.2f')
```

### Visualizações Geradas

- Histogramas de todas as variáveis numéricas
- Boxplots por grupo (doente vs. saudável)
- Heatmap de correlação
- Distribuição de features categóricas (barplot)
- Curvas de densidade por target

---

## Etapa 3 — Pré-processamento

### 3.1 Limpeza de Dados (dataset UCI)

```python
# Remover coluna ID (sem valor preditivo)
df = df.drop(columns=['id'])

# Converter booleanos texto para numérico
df['fbs'] = df['fbs'].map({'TRUE': 1, 'FALSE': 0})
df['exang'] = df['exang'].map({'TRUE': 1, 'FALSE': 0})

# Binarizar target (num 0–4 → 0/1)
df['target'] = (df['num'] > 0).astype(int)
df = df.drop(columns=['num'])
```

### 3.2 Tratamento de Valores Ausentes

```python
from sklearn.impute import KNNImputer

# Colunas com missing para imputação KNN
cols_knn = ['ca', 'slope', 'thalch', 'chol', 'trestbps']

# Imputar colunas categóricas (thal, fbs) por moda por grupo
df['thal'] = df.groupby('dataset')['thal'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.median())
)

# KNN Imputer para numéricas
imputer = KNNImputer(n_neighbors=5)
df[cols_knn] = imputer.fit_transform(df[cols_knn])
```

### 3.3 Encoding de Variáveis Categóricas

```python
# Converter sexo para binário
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# Encoding ordinal para cp (mapeamento médico)
cp_mapping = {
    'typical angina': 3,
    'atypical angina': 1,
    'non-anginal': 2,
    'asymptomatic': 0
}
df['cp'] = df['cp'].map(cp_mapping)

# Encoding ordinal para thal
thal_mapping = {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
df['thal'] = df['thal'].map(thal_mapping)

# Encoding ordinal para slope
slope_mapping = {'upsloping': 2, 'flat': 1, 'downsloping': 0}
df['slope'] = df['slope'].map(slope_mapping)

# Encoding ordinal para restecg
restecg_mapping = {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2}
df['restecg'] = df['restecg'].map(restecg_mapping)

# Drop coluna dataset (não disponível em produção)
df = df.drop(columns=['dataset'])
```

### 3.4 Feature Engineering

```python
# Grupo etário
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 65, 100],
                          labels=['young', 'middle', 'senior', 'elderly'])
df['age_group'] = df['age_group'].cat.codes  # ordinal encoding

# Ratios clínicos
df['chol_age_ratio'] = df['chol'] / df['age']
df['thalach_age_ratio'] = df['thalach'] / df['age']

# Índice de severidade do segmento ST
df['st_severity'] = df['oldpeak'] * (2 - df['slope'])

# Score de risco composto
df['risk_score'] = (df['age'] > 55).astype(int) + \
                   df['sex'] + \
                   (df['cp'] == 0).astype(int) + \
                   df['exang']
```

### 3.5 Normalização

```python
from sklearn.preprocessing import StandardScaler

# Features numéricas contínuas para normalizar
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                'chol_age_ratio', 'thalach_age_ratio', 'st_severity']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
```

---

## Etapa 4 — Divisão dos Dados

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

# Split estratificado: 70% treino, 15% validação, 15% teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_test.shape[0]}")
```

**Estratificação:** garante proporção igual de target=0/1 em todos os subconjuntos.

---

## Etapa 5 — Treinamento

### Configuração do Experimento MLflow

```python
import mlflow

mlflow.set_experiment("heart-disease-experiment")
mlflow.autolog(disable=False)  # Para Random Forest

# Ou logging manual para XGBoost/LightGBM
with mlflow.start_run(run_name="xgboost_run"):
    ...
```

### Validação Cruzada Estratificada

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_auc = cross_val_score(
    model, X_train, y_train,
    cv=cv, scoring='roc_auc'
)
cv_scores_f1 = cross_val_score(
    model, X_train, y_train,
    cv=cv, scoring='f1'
)

print(f"CV AUC-ROC: {cv_scores_auc.mean():.4f} ± {cv_scores_auc.std():.4f}")
```

### Otimização com Optuna

```python
import optuna

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_auc = study.best_value
```

---

## Etapa 6 — Avaliação

### Métricas Calculadas

```python
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve
)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'auc_roc': roc_auc_score(y_test, y_proba),
    'f1': f1_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred)
}
```

### Critério de Seleção

| Critério | Limiar | Justificativa |
|----------|--------|--------------|
| AUC-ROC | ≥ 0.85 | Padrão para diagnóstico clínico auxiliar |
| Recall | ≥ 0.80 | Falso negativo (miss) é mais custoso que falso positivo |
| F1-Score | ≥ 0.82 | Balanceamento geral |

---

## Etapa 7 — Registro do Modelo

```python
import mlflow.lightgbm

# Registrar o melhor modelo
with mlflow.start_run(run_name="best_model_registration"):
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    mlflow.lightgbm.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="heart-disease-best-model"
    )
```

No portal Azure ML:
1. Acesse **Models** no workspace
2. O modelo `heart-disease-best-model` estará disponível com versão, tags e link ao run de origem

---

## Fluxo Resumido

```
CSV (local / Blob) 
      ↓
EDA + Limpeza
      ↓
Encoding + Feature Engineering
      ↓
StandardScaler
      ↓
Train / Val / Test Split (70/15/15, estratificado)
      ↓
GridSearchCV / Optuna
      ↓
Cross-validation (5-fold, estratificado)
      ↓
Avaliação no conjunto de teste
      ↓
MLflow logging (params + metrics + artefatos)
      ↓
SHAP analysis
      ↓
Model Registry (melhor modelo)
```
