# Dicionário de Dados

*Última atualização: Maio 2026*

---

## Datasets Disponíveis

| Arquivo | Registros | Features | Target | Uso |
|---------|-----------|---------|--------|-----|
| `heart.csv` | 1.025 | 13 | `target` (0/1) | Notebook de referência |
| `heart_disease_uci.csv` | 920 | 14 | `num` (0–4) | Notebook avançado |

---

## heart.csv

Versão simplificada e limpa do dataset Heart Disease Cleveland (UCI), amplamente utilizado em benchmarks de ML.

**Fonte:** [Heart Disease Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
**Distribuição do target:** 526 positivos (51,3%) / 499 negativos (48,7%)

### Dicionário de Variáveis

| Variável | Tipo | Descrição | Valores / Range | Observações |
|----------|------|-----------|----------------|-------------|
| `age` | Inteiro | Idade do paciente em anos | 29 – 77 | Pacientes com ≥ 55 anos têm maior risco |
| `sex` | Binário | Sexo biológico | 0 = feminino, 1 = masculino | Homens representam ~68% do dataset |
| `cp` | Categórico (0–3) | Tipo de dor torácica (chest pain) | 0 = assintomático, 1 = angina atípica, 2 = dor não-anginosa, 3 = angina típica | Fortemente associado ao target |
| `trestbps` | Inteiro | Pressão arterial em repouso (mmHg) | 94 – 200 | Hipertensão: > 140 mmHg |
| `chol` | Inteiro | Colesterol sérico (mg/dl) | 126 – 564 | Elevado: > 240 mg/dl |
| `fbs` | Binário | Glicemia em jejum > 120 mg/dl | 0 = falso, 1 = verdadeiro | Indicador de diabetes |
| `restecg` | Categórico (0–2) | Resultado do ECG em repouso | 0 = normal, 1 = anormalidade ST-T, 2 = hipertrofia ventricular esquerda | |
| `thalach` | Inteiro | Frequência cardíaca máxima atingida | 71 – 202 bpm | Correlação negativa com doença |
| `exang` | Binário | Angina induzida por exercício | 0 = não, 1 = sim | |
| `oldpeak` | Float | Depressão do segmento ST induzida por exercício em relação ao repouso | 0.0 – 6.2 | Valores altos indicam isquemia |
| `slope` | Categórico (0–2) | Inclinação do segmento ST no pico do exercício | 0 = descida (downsloping), 1 = plano (flat), 2 = subida (upsloping) | |
| `ca` | Inteiro (0–3) | Número de vasos principais coloridos por fluoroscopia | 0, 1, 2, 3 | Importante preditor: mais vasos → maior risco |
| `thal` | Categórico (0–3) | Tipo de talassemia | 0 = normal, 1 = defeito fixo, 2 = defeito reversível, 3 = sem resultado | `thal=2` associado a maior risco |
| `target` | **Binário** | **Variável-alvo: presença de doença cardíaca** | **0 = ausente, 1 = presente** | **Variável a ser predita** |

---

## heart_disease_uci.csv

Dataset original do UCI com dados de 4 centros clínicos internacionais.

**Fonte:** [UCI ML Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)  
**Centros clínicos:** Cleveland (EUA), Hungary (Hungria), Switzerland (Suíça), VA Long Beach (EUA)  
**Distribuição do target:** `num=0` (saudável: 45%) / `num≥1` (doença: 55%)

### Diferenças em relação ao heart.csv

| Característica | heart.csv | heart_disease_uci.csv |
|----------------|-----------|----------------------|
| Centros clínicos | Cleveland | Cleveland + 3 centros |
| Registros | 1.025 | 920 |
| Coluna target | `target` (0/1) | `num` (0–4) → binarizado |
| Valores ausentes | Nenhum | Presentes em `ca`, `thal`, `slope` |
| Nome da feature `thalach` | `thalach` | `thalch` |
| Centros identificados | Não | Coluna `dataset` |

### Colunas Adicionais

| Variável | Tipo | Descrição | Valores |
|----------|------|-----------|---------|
| `id` | Inteiro | Identificador único do paciente | 1 – 920 |
| `dataset` | String | Centro clínico de origem | Cleveland, Hungary, Switzerland, VA Long Beach |
| `num` | Inteiro (0–4) | Grau de obstrução arterial | 0 = sem doença; 1–4 = grau crescente de obstrução |

### Binarização do Target no Notebook Avançado

```python
df['target'] = (df['num'] > 0).astype(int)
```

### Valores Ausentes

| Coluna | Ausentes | Estratégia de Imputação |
|--------|----------|------------------------|
| `ca` | ~67 | KNN Imputer (k=5) |
| `thal` | ~2 | Moda por centro clínico |
| `slope` | ~309 | KNN Imputer (k=5) |
| `thalch` | ~55 | KNN Imputer (k=5) |
| `fbs` | ~90 | Mediana por sexo e faixa etária |
| `chol` | ~30 | KNN Imputer (k=5) |
| `trestbps` | ~59 | KNN Imputer (k=5) |

---

## Features Derivadas (Feature Engineering)

Criadas no notebook `treinamento_modelos_ml.ipynb`:

| Feature | Fórmula | Significado Médico |
|---------|---------|-------------------|
| `age_group` | Faixa etária categórica | Estratificação por risco etário |
| `chol_age_ratio` | `chol / age` | Colesterol relativo à idade |
| `thalach_age_ratio` | `thalach / age` | Capacidade cardíaca relativa à idade |
| `st_severity` | `oldpeak × slope_numeric` | Índice combinado de alteração ST |
| `risk_score` | Combinação de `age`, `sex`, `cp`, `exang` | Score de risco composto |

---

## Estatísticas Descritivas (heart.csv)

| Variável | Média | Desvio Padrão | Min | Mediana | Max |
|----------|-------|--------------|-----|---------|-----|
| `age` | 54.4 | 9.1 | 29 | 55 | 77 |
| `trestbps` | 131.6 | 17.5 | 94 | 130 | 200 |
| `chol` | 246.0 | 51.6 | 126 | 240 | 564 |
| `thalach` | 149.6 | 22.9 | 71 | 153 | 202 |
| `oldpeak` | 1.07 | 1.16 | 0.0 | 0.8 | 6.2 |

---

## Correlações com o Target

Features com maior correlação (valor absoluto de Pearson) com `target`:

| Feature | Correlação | Direção |
|---------|-----------|---------|
| `cp` | 0.43 | Positiva (cp=0 → doença) |
| `thalach` | 0.42 | Negativa (mais bpm → saudável) |
| `exang` | 0.44 | Positiva (angina → doença) |
| `oldpeak` | 0.43 | Positiva (depressão ST → doença) |
| `ca` | 0.46 | Positiva (mais vasos → doença) |
| `thal` | 0.52 | Positiva (def. reversível → doença) |

---

## Referências

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology*, 64(5), 304-310.
