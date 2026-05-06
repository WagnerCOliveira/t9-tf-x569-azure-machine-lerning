# Arquitetura do Projeto — Azure Machine Learning

*Última atualização: Maio 2026*

---

## Visão Geral

O projeto segue uma arquitetura de ML em nuvem com **Azure Machine Learning** como plataforma central, utilizando componentes gerenciados para computação, armazenamento e rastreamento de experimentos. O design prioriza reprodutibilidade, rastreabilidade e custo controlado.

![Arquitetura Azure ML](arquitetura_azure.png)

---

## Componentes do Azure ML

### 1. Resource Group — `rg-projeto-final-ml`

Contêiner lógico que agrupa todos os recursos do projeto na região `eastus`. Facilita:
- Controle de custos consolidado
- Deleção completa do ambiente com um único comando
- Organização por projeto no portal Azure

### 2. AML Workspace — `mlw-projeto-final`

Hub central do experimento. Criado com a extensão `az ml`, provisiona automaticamente:
- **Azure Storage Account** — armazenamento de datasets, artefatos e modelos
- **Azure Key Vault** — gerenciamento seguro de credenciais e segredos
- **Azure Application Insights** — telemetria e monitoramento de endpoints
- **Azure Container Registry** — imagens Docker dos ambientes de treinamento

### 3. Compute Instance — `ci-notebooks-trabalho`

VM dedicada para desenvolvimento interativo via Jupyter:

| Atributo | Valor |
|----------|-------|
| Tipo | `STANDARD_DS11_V2` |
| vCPUs | 2 |
| RAM | 14 GB |
| Sistema Operacional | Ubuntu 20.04 LTS |
| Kernels disponíveis | Python 3.8/3.10 — AzureML |

**Uso:** Execução dos notebooks de EDA e treinamento. Deve ser **parada manualmente** quando não utilizada para evitar cobranças.

### 4. Compute Cluster — `cluster-treino-trabalho`

Cluster escalável para treinos mais intensivos:

| Atributo | Valor |
|----------|-------|
| Tipo | `AmlCompute` |
| VM Size | `STANDARD_DS11_V2` |
| Mínimo de nós | 0 (sem custo quando ocioso) |
| Máximo de nós | 2 |
| Idle secs before scale down | 120 |

**Decisão de design:** `min-instances=0` garante que o cluster não gere custos quando não há jobs em execução. O scale-up ocorre automaticamente ao submeter um job.

### 5. Datastore — `workspaceblobstore`

Container padrão do Azure Blob Storage criado automaticamente com o workspace. Utilizado para:
- Armazenar os arquivos CSV de dados de entrada
- Persistir artefatos de modelos (curvas ROC, matrizes de confusão)
- Manter checkpoints de execução

**Path de acesso nos notebooks:**
```python
path = "azureml://datastores/workspaceblobstore/paths/heart.csv"
```

### 6. MLflow Experiment — `heart-disease-experiment`

Rastreamento de todas as runs de treinamento com:
- **Hiperparâmetros** registrados automaticamente (autolog) ou manualmente
- **Métricas** de treino e validação por época/fold
- **Artefatos** — curvas ROC (PNG), matrizes de confusão, feature importance
- **Tags** de metadados (modelo, dataset, versão)

**Runs registradas:**
- `random_forest_run` — Random Forest com autolog MLflow
- `xgboost_run` — XGBoost com logging manual
- Runs adicionais do notebook avançado (LightGBM, Logistic Regression, Ensemble, Optuna)

### 7. Model Registry

Após a seleção do melhor modelo, ele é registrado no Azure ML Model Registry com:
- Nome versionado: `heart-disease-best-model`
- Tags de performance (AUC, F1)
- Referência ao run de origem

---

## Fluxo de Dados End-to-End

```
[Arquivo Local]           [Azure Blob Storage]
 data/heart.csv    ──→    workspaceblobstore
 data/heart_disease_uci.csv

         ↓
[Compute Instance — Notebook]
  1. Leitura via azureml:// path (ou fallback local)
  2. EDA e Pré-processamento
  3. Feature Engineering
  4. Split train/validation/test

         ↓
[Treinamento — Compute Instance ou Cluster]
  5. GridSearchCV / Optuna
  6. Cross-validation estratificado
  7. Avaliação de métricas

         ↓
[MLflow Tracking]
  8. Log de parâmetros, métricas e artefatos
  9. Comparação de runs no portal ml.azure.com

         ↓
[Model Registry]
  10. Registro do melhor modelo com tags
  11. Versionamento para reprodução futura
```

---

## Decisões de Design

### Fallback Local Automático

O notebook detecta se está rodando em ambiente Azure ML:
```python
try:
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    df = pd.read_csv("azureml://datastores/workspaceblobstore/paths/heart.csv")
except:
    df = pd.read_csv("../data/heart.csv")  # fallback local
```
**Motivo:** Permite desenvolvimento e testes locais sem necessidade de acesso à nuvem a todo momento, reduzindo custos durante iterações.

### Sem Pipeline Formal do Azure ML

O projeto usa **notebooks interativos** em vez de `azure.ai.ml.dsl.pipeline`. 
**Motivo:** Para fins acadêmicos, notebooks oferecem maior visibilidade e facilidade de apresentação das etapas. Pipelines formais seriam recomendados em cenário de produção com CI/CD.

### MLflow como Backend de Tracking

Preferência ao MLflow em vez do tracking nativo do AML.
**Motivo:** Portabilidade — o mesmo código funciona com tracking local (`mlflow.set_tracking_uri("./mlruns")`) ou conectado ao AML, facilitando desenvolvimento local.

### VM Size STANDARD_DS11_V2

Escolha motivada pelo equilíbrio entre:
- Disponibilidade na região `eastus` com créditos estudantis
- Memória suficiente (14 GB) para datasets de até ~500 MB com pandas
- Custo por hora compatível com orçamento acadêmico

---

## Diagrama de Componentes

```
┌─────────────────────────────────────────────────────┐
│              Azure Resource Group                    │
│              rg-projeto-final-ml                     │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │         AML Workspace: mlw-projeto-final      │    │
│  │                                               │    │
│  │  ┌──────────────┐  ┌──────────────────────┐  │    │
│  │  │   Compute     │  │   Blob Storage       │  │    │
│  │  │   Instance    │  │   workspaceblobstore  │  │    │
│  │  │  (Notebooks)  │  │   heart.csv          │  │    │
│  │  └──────┬───────┘  └──────────────────────┘  │    │
│  │         │                                     │    │
│  │  ┌──────▼───────┐  ┌──────────────────────┐  │    │
│  │  │   MLflow      │  │   Model Registry     │  │    │
│  │  │   Tracking    │  │   (best model v1)    │  │    │
│  │  │   Experiment  │  └──────────────────────┘  │    │
│  │  └──────────────┘                             │    │
│  │                                               │    │
│  │  ┌──────────────┐  ┌──────────────────────┐  │    │
│  │  │  Compute      │  │   Key Vault          │  │    │
│  │  │  Cluster      │  │   App Insights       │  │    │
│  │  │  (0-2 nós)    │  │   Container Registry │  │    │
│  │  └──────────────┘  └──────────────────────┘  │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## Referências

- [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning/)
- [MLflow on Azure ML](https://learn.microsoft.com/azure/machine-learning/how-to-use-mlflow-cli-runs)
- [AML Compute Targets](https://learn.microsoft.com/azure/machine-learning/concept-compute-target)
