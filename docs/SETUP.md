# Guia de Configuração do Ambiente

*Última atualização: Maio 2026*

---

## Pré-requisitos

| Requisito | Versão Mínima | Como Verificar |
|-----------|--------------|----------------|
| Python | 3.10 | `python --version` |
| Azure CLI | 2.50+ | `az --version` |
| Git | 2.x | `git --version` |
| Conta Azure | — | [portal.azure.com](https://portal.azure.com) |

---

## 1. Configuração Local

### 1.1 Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd projeto
```

### 1.2 Criar Ambiente Virtual (recomendado)

```bash
# Usando venv
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Ou usando conda
conda create -n heart-ml python=3.10
conda activate heart-ml
```

### 1.3 Instalar Dependências Locais

```bash
pip install -r requirements-local.txt
```

### 1.4 Verificar a Instalação

```bash
python -c "import sklearn, xgboost, lightgbm, optuna, shap; print('OK')"
```

### 1.5 Executar Notebooks Localmente

```bash
jupyter lab
# ou
jupyter notebook
```

Abra `notebooks/treinamento_modelos_ml.ipynb`. O notebook detecta automaticamente que não está no Azure e usa fallback local.

---

## 2. Configuração do Azure

### 2.1 Instalar Azure CLI

**Ubuntu/Debian:**
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

**Windows (via PowerShell):**
```powershell
winget install Microsoft.AzureCLI
```

**macOS:**
```bash
brew install azure-cli
```

### 2.2 Fazer Login na Azure

```bash
az login
```

Um browser abrirá para autenticação. Após login:
```bash
# Verificar assinatura ativa
az account show

# Listar assinaturas disponíveis
az account list --output table

# Selecionar assinatura específica (se necessário)
az account set --subscription "<nome-ou-id>"
```

### 2.3 Criar o Ambiente Azure ML

Execute o script de inicialização na raiz do projeto:

```bash
bash scripts/start-environment.sh
```

O script criará automaticamente:
- Resource Group `rg-projeto-final-ml` (região `eastus`)
- Workspace `mlw-projeto-final`
- Compute Instance `ci-notebooks-trabalho` (STANDARD_DS11_V2)
- Compute Cluster `cluster-treino-trabalho` (0–2 nós)

**Tempo estimado:** 5–10 minutos

> Aguarde a mensagem `AMBIENTE PRONTO!` antes de prosseguir.

### 2.4 Fazer Upload do Dataset

**Opção A — Portal Web:**
1. Acesse [ml.azure.com](https://ml.azure.com)
2. Selecione o workspace `mlw-projeto-final`
3. Menu lateral → **Data** → **Datastores**
4. Clique em `workspaceblobstore` → **Browse** → **Upload**
5. Selecione `data/heart.csv`

**Opção B — Azure CLI:**
```bash
# Obter nome da storage account
az ml workspace show --name mlw-projeto-final -g rg-projeto-final-ml \
  --query storageAccount -o tsv

# Upload do arquivo
az storage blob upload \
  --account-name <storage_account_name> \
  --container-name azureml-blobstore-<id> \
  --name heart.csv \
  --file data/heart.csv \
  --auth-mode login
```

### 2.5 Instalar Dependências na Compute Instance

1. Acesse [ml.azure.com](https://ml.azure.com) → **Compute** → **ci-notebooks-trabalho** → **Terminal**
2. Execute:
```bash
pip install -r requirements-azure.txt
```

### 2.6 Executar os Notebooks no Azure ML

1. Acesse [ml.azure.com](https://ml.azure.com) → **Notebooks**
2. Clique em **Upload files** → selecione o notebook desejado
3. Selecione a Compute Instance `ci-notebooks-trabalho`
4. Selecione kernel **Python 3.10**
5. Execute **Run All** (ou célula por célula)

---

## 3. Variáveis de Ambiente (opcional)

Para uso da autenticação service principal em pipelines automatizados:

```bash
export AZURE_CLIENT_ID="<id-do-service-principal>"
export AZURE_CLIENT_SECRET="<secret>"
export AZURE_TENANT_ID="<tenant-id>"
export AZURE_SUBSCRIPTION_ID="<subscription-id>"
```

> **Nunca** commite variáveis de ambiente com credenciais no repositório. Use o arquivo `.env` (incluído no `.gitignore`) para desenvolvimento local.

---

## 4. Limpeza do Ambiente

Para remover todos os recursos Azure e encerrar cobranças:

```bash
bash scripts/end-environment.sh
```

> **Atenção:** Remove o Resource Group e **todos** os recursos de forma irreversível, incluindo modelos registrados e datasets no Blob Storage. Faça backup local antes.

---

## 5. Troubleshooting

### Erro: `az ml: command not found`

A extensão Azure ML não está instalada. Execute:
```bash
az extension add -n ml -y
```

### Erro: `DefaultAzureCredential failed`

Execute no terminal:
```bash
az login
az account set --subscription "<sua-subscription>"
```

Ou no notebook:
```python
from azure.identity import AzureCliCredential
credential = AzureCliCredential()
```

### Erro: `ModuleNotFoundError: No module named 'lightgbm'`

```bash
pip install lightgbm>=4.0.0
```

Se estiver na Compute Instance:
```bash
pip install --user lightgbm>=4.0.0
```
Reinicie o kernel após instalar.

### Erro: Dataset não encontrado no Blob Storage

Verifique se o upload foi feito corretamente:
```bash
az storage blob list \
  --account-name <storage_account> \
  --container-name azureml-blobstore-<id> \
  --auth-mode login \
  --output table
```

### Compute Instance não inicia

Verifique se há cotas disponíveis na assinatura:
```bash
az vm list-usage --location eastus --output table | grep DSv2
```

---

## 6. Custos Estimados

| Recurso | Tipo | Custo Estimado |
|---------|------|---------------|
| Workspace | Fixo | ~$0/mês (sem execuções) |
| Compute Instance | Por hora de uso | ~$0.22/hora (DS11_v2) |
| Compute Cluster (ocioso) | Por hora | $0 (min-instances=0) |
| Blob Storage | Por GB | ~$0.018/GB/mês |

**Recomendação:** Pare a Compute Instance no portal quando não estiver usando (`Compute` → `ci-notebooks-trabalho` → **Stop**).

---

## Referências

- [Instalar Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
- [Azure ML Quickstart](https://learn.microsoft.com/azure/machine-learning/quickstart-create-resources)
- [DefaultAzureCredential](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential)
