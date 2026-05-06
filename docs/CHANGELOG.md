# Changelog

Todas as mudanças relevantes deste projeto são documentadas aqui.

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

---

## [1.2.0] — 2026-05-06

### Adicionado
- Notebook `treinamento_modelos_ml.ipynb` com pipeline completo de 4 modelos
- Otimização bayesiana com Optuna (50 trials, TPE sampler) para LightGBM
- Análise de interpretabilidade SHAP (beeswarm, bar plot, waterfall)
- Voting Ensemble (soft voting) combinando LR + LightGBM + XGBoost
- Feature engineering com 5 novas variáveis derivadas
- Documentação completa em `/docs` (ARCHITECTURE, DATA_DICTIONARY, MODELS, PIPELINE, SETUP, CHANGELOG)
- README.md expandido com badges, tabelas de performance e índice

### Corrigido
- Grid da Logistic Regression corrigido (incompatibilidade solver/penalty)
- `predict_proba` do hard voting corrigido para compatibilidade com scikit-learn 1.5

---

## [1.1.0] — 2025-11-10

### Adicionado
- Notebook `data processed.ipynb` com EDA completo do dataset UCI
- Dataset `heart_disease_uci.csv` (920 registros, 4 centros clínicos)
- Imputação ML-based (KNN Imputer) para valores ausentes do dataset UCI
- Seção de feature engineering no notebook avançado

### Alterado
- `requirements-azure.txt` atualizado para `scikit-learn==1.5.1`
- `README.md` expandido com pipeline completo e tabela de resultados UCI

### Removido
- `numpy` do `requirements-azure.txt` (pré-instalado na Compute Instance)

---

## [1.0.0] — 2025-10-15

### Adicionado
- Notebook inicial `trabalho_final_ml.ipynb` com Random Forest e XGBoost
- Pipeline completo: Blob Storage → EDA → treinamento → MLflow logging
- Scripts de automação `start-environment.sh` e `end-environment.sh`
- Dataset `heart.csv` (1.025 registros, versão Cleveland)
- `requirements-azure.txt` e `requirements-local.txt`
- `README.md` inicial com instruções de setup
- Diagramas de arquitetura e pipeline em `docs/`

---

## Tipos de Mudança

- **Adicionado** — nova funcionalidade
- **Alterado** — mudança em funcionalidade existente
- **Descontinuado** — funcionalidade a ser removida em versão futura
- **Removido** — funcionalidade removida
- **Corrigido** — correção de bug
- **Segurança** — correção de vulnerabilidade
