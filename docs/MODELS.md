# Documentação dos Modelos

*Última atualização: Maio 2026*

---

## Visão Geral

O projeto implementa quatro algoritmos de classificação binária para previsão de doenças cardíacas. A seleção cobre o espectro de modelos lineares simples a ensembles complexos, permitindo comparação rigorosa de performance e interpretabilidade.

| Modelo | Tipo | AUC-ROC | F1-Score | Complexidade |
|--------|------|---------|---------|--------------|
| Logistic Regression | Linear | ~0.88 | ~0.85 | Baixa |
| LightGBM (Grid) | Gradient Boosting | ~0.89 | ~0.86 | Média |
| XGBoost | Gradient Boosting | ~0.90 | ~0.87 | Média-Alta |
| Voting Ensemble | Ensemble | ~0.90 | ~0.87 | Alta |
| LightGBM + Optuna | Gradient Boosting | ~0.91 | ~0.88 | Alta |

---

## 1. Logistic Regression (Baseline)

### Descrição

Modelo linear que estima a probabilidade de `target=1` usando a função sigmoide aplicada a uma combinação linear das features. Serve como baseline interpretável.

### Hiperparâmetros Utilizados

```python
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'max_iter': [200, 500]
}
```

**`C` (inverso da regularização):** Controla overfitting. Valores menores → mais regularização.  
**`penalty`:** L1 produz esparsidade (seleção de features), L2 distribui o peso, ElasticNet combina ambas.

### Métricas no Conjunto de Teste

| Métrica | Valor |
|---------|-------|
| AUC-ROC | ~0.88 |
| F1-Score | ~0.85 |
| Accuracy | ~0.84 |
| Precision | ~0.87 |
| Recall | ~0.84 |

### Interpretação dos Coeficientes

Features com coeficientes mais relevantes (sinais alinhados com evidência médica):

| Feature | Direção | Interpretação |
|---------|---------|--------------|
| `thal` | Positivo | Talassemia reversível aumenta risco |
| `ca` | Positivo | Mais vasos obstruídos aumenta risco |
| `exang` | Positivo | Angina por exercício indica doença |
| `thalach` | Negativo | Maior frequência máxima indica saúde |
| `cp` (asymptomatic) | Positivo | Paradoxalmente associado a maior risco |

### Quando Utilizar

- Quando interpretabilidade dos coeficientes é mandatória (relatórios médicos)
- Como baseline para comparação com modelos mais complexos
- Quando o dataset é pequeno e há risco de overfitting com boosting

### Limitações

- Assume linearidade entre features e log-odds do target
- Sensível a features correlacionadas (multicolinearidade)
- Não captura interações não-lineares automaticamente

---

## 2. LightGBM

### Descrição

Gradient boosting framework da Microsoft otimizado para velocidade e eficiência de memória. Usa histogramas para aproximar divisões de árvore, sendo ~6× mais rápido que XGBoost em datasets grandes.

### Configuração — GridSearchCV

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [15, 31, 63],
    'min_child_samples': [20, 50]
}
```

### Configuração — Optuna (50 trials, TPE Sampler)

```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    ...
```

**TPE (Tree-structured Parzen Estimator):** Sampler bayesiano que modela a distribuição dos bons hiperparâmetros e direciona a busca para regiões promissoras.

### Métricas no Conjunto de Teste

| Configuração | AUC-ROC | F1-Score |
|-------------|---------|---------|
| GridSearchCV | ~0.89 | ~0.86 |
| Optuna (50 trials) | ~0.91 | ~0.88 |

### Comparação com XGBoost

| Critério | LightGBM | XGBoost |
|----------|---------|---------|
| Velocidade de treino | ✅ Mais rápido | Mais lento |
| Memória | ✅ Menor uso | Maior uso |
| AUC-ROC (sem tuning) | 0.89 | 0.90 |
| AUC-ROC (com Optuna) | **0.91** | — |
| Datasets pequenos | ⚠️ Pode overfitting | Mais robusto |
| Interpretabilidade SHAP | Nativa | Nativa |

### Casos de Uso Recomendados

- Datasets > 10.000 registros onde velocidade importa
- Cenários com restrições de memória
- Quando se planeja otimização bayesiana extensiva

---

## 3. XGBoost

### Descrição

Gradient boosting com regularização L1/L2 explícita. Considerado o algoritmo padrão para dados tabulares em competições ML. Utiliza crescimento de árvore por nível (level-wise), diferente do LightGBM que usa leaf-wise.

### Hiperparâmetros Utilizados

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}
```

### Métricas no Conjunto de Teste

| Métrica | Valor |
|---------|-------|
| AUC-ROC | ~0.90 |
| F1-Score | ~0.87 |
| Accuracy | ~0.86 |

### Feature Importance (gain)

Top features por importância média (gain):

| Rank | Feature | Importância Relativa |
|------|---------|---------------------|
| 1 | `thal` | Alta |
| 2 | `cp` | Alta |
| 3 | `ca` | Alta |
| 4 | `thalach` | Média-Alta |
| 5 | `oldpeak` | Média |
| 6 | `age` | Média |
| 7 | `exang` | Média |

### Logging no MLflow

O XGBoost usa **logging manual** no projeto (em contraste com o Random Forest que usa autolog):
```python
with mlflow.start_run(run_name="xgboost_run"):
    mlflow.log_params(best_params)
    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_artifact("roc_curve_xgb.png")
    mlflow.xgboost.log_model(model, "model")
```

---

## 4. Voting Ensemble (Soft Voting)

### Descrição

Combina as predições de probabilidade dos três modelos treinados usando média ponderada (soft voting). Reduz variância e melhora estabilidade em relação a modelos individuais.

### Composição

```python
estimators = [
    ('lr', logistic_regression_best),
    ('lgbm', lightgbm_best),
    ('xgb', xgboost_best)
]

voting_clf = VotingClassifier(
    estimators=estimators,
    voting='soft',  # usa probabilidades, não votos binários
    weights=[1, 2, 2]  # LightGBM e XGBoost têm peso 2x
)
```

**Soft voting vs Hard voting:**
- **Soft:** `P(target=1) = mean([P_lr, P_lgbm, P_xgb] × weights)` — recomendado quando os modelos são calibrados
- **Hard:** voto majoritário — mais adequado quando calibração não é garantida

### Pesos dos Estimadores

| Estimador | Peso | Justificativa |
|-----------|------|--------------|
| Logistic Regression | 1 | Modelo linear, menor poder preditivo |
| LightGBM | 2 | Alto AUC, bom poder preditivo |
| XGBoost | 2 | Melhor AUC individual |

### Performance Comparada

| Configuração | AUC-ROC | F1-Score |
|-------------|---------|---------|
| LightGBM individual | 0.89 | 0.86 |
| XGBoost individual | 0.90 | 0.87 |
| Voting Ensemble | 0.90 | 0.87 |

O ensemble iguala o melhor modelo individual e apresenta maior robustez (menor variância em validação cruzada).

---

## Análise SHAP

### Metodologia

SHAP (SHapley Additive exPlanations) quantifica a contribuição de cada feature para cada predição individual, baseado na teoria dos jogos cooperativos de Shapley.

### Visualizações Geradas

**Beeswarm Plot** — Distribuição dos valores SHAP por feature:
- Eixo X: magnitude do impacto na predição
- Cor: valor da feature (vermelho=alto, azul=baixo)
- Cada ponto = um paciente

**Bar Plot** — Importância média absoluta:
- Ranking global de features por impacto

**Waterfall Plot** — Explicação individual:
- Mostra como cada feature contribuiu para uma predição específica
- Gerado para True Positives e False Negatives para análise de erros

### Principais Insights

1. **`thal=reversable defect`** é o preditor mais forte: pacientes com esse padrão têm alta probabilidade de doença
2. **`cp=asymptomatic`** — contra-intuitivamente associado a maior risco (isquemia silenciosa)
3. **`ca`** (vasos obstruídos): cada vaso adicional colorido aumenta o risco significativamente
4. **`thalach` baixo** em pacientes mais velhos indica baixa reserva cardíaca
5. **Falsos negativos** concentram-se em perfis: `age<50`, `cp=typical angina`, `thal=normal` — pacientes jovens com sintomas típicos

---

## Critério de Seleção do Melhor Modelo

O modelo registrado no Azure ML Model Registry é selecionado por:

1. **AUC-ROC** como métrica primária (≥ 0.85 obrigatório)
2. **F1-Score** como métrica secundária (balanceamento precision/recall)
3. **Recall** priorizado sobre Precision no contexto médico (custo de falso negativo > falso positivo)

**Modelo selecionado:** LightGBM + Optuna com AUC-ROC ≈ 0.91
