#!/bin/bash
# =============================================================================
# Script: start-environment.sh
# Descrição: Cria todos os recursos Azure necessários para o projeto de ML
#            (Resource Group, Workspace, Compute Instance e Compute Cluster)
# Autor: Wagner Costa Oliveira
# Data: 2025-10-15
# Uso: bash scripts/start-environment.sh
# Pré-requisito: az login executado previamente com conta com créditos ativos
# Tempo estimado: 5–10 minutos
# =============================================================================

set -e  # Abortar em caso de erro em qualquer comando

# --- Configuração de Variáveis ---
RG_NAME="rg-projeto-final-ml"
LOCATION="eastus"
WS_NAME="mlw-projeto-final"
CI_NAME="ci-notebooks-trabalho"
CLUSTER_NAME="cluster-treino-trabalho"
VM_SIZE="STANDARD_DS11_V2"

echo "============================================================"
echo " INICIANDO CRIAÇÃO DO AMBIENTE AZURE ML"
echo "============================================================"
echo " Resource Group : $RG_NAME"
echo " Região         : $LOCATION"
echo " Workspace      : $WS_NAME"
echo " Compute Inst.  : $CI_NAME ($VM_SIZE)"
echo " Cluster        : $CLUSTER_NAME ($VM_SIZE, 0-2 nós)"
echo "============================================================"

# --- Instalar/Atualizar extensão Azure ML ---
echo ""
echo "[1/5] Instalando extensão Azure ML (az ml)..."
az extension add -n ml -y 2>/dev/null || az extension update -n ml -y
echo "      Extensão instalada com sucesso."

# --- Criar Resource Group ---
echo ""
echo "[2/5] Criando Resource Group: $RG_NAME..."
az group create --name "$RG_NAME" --location "$LOCATION" --output none
echo "      Resource Group criado: $RG_NAME ($LOCATION)"

# --- Criar Workspace Azure ML ---
echo ""
echo "[3/5] Criando Workspace: $WS_NAME..."
echo "      (isso leva cerca de 3–5 minutos — aguarde...)"
az ml workspace create \
  --name "$WS_NAME" \
  --resource-group "$RG_NAME" \
  --location "$LOCATION" \
  --output none
echo "      Workspace criado: $WS_NAME"

# --- Criar Compute Instance (para notebooks interativos) ---
echo ""
echo "[4/5] Criando Compute Instance: $CI_NAME..."
echo "      Tipo: $VM_SIZE (2 vCPUs, 14 GB RAM)"
az ml compute create \
  --name "$CI_NAME" \
  --size "$VM_SIZE" \
  --type ComputeInstance \
  --workspace-name "$WS_NAME" \
  --resource-group "$RG_NAME" \
  --output none
echo "      Compute Instance criada: $CI_NAME"

# --- Criar Compute Cluster (para treinos pesados) ---
echo ""
echo "[5/5] Criando Compute Cluster: $CLUSTER_NAME..."
echo "      Configurado com min-instances=0 para não gerar custos quando ocioso."
az ml compute create \
  --name "$CLUSTER_NAME" \
  --size "$VM_SIZE" \
  --min-instances 0 \
  --max-instances 2 \
  --type AmlCompute \
  --workspace-name "$WS_NAME" \
  --resource-group "$RG_NAME" \
  --output none
echo "      Compute Cluster criado: $CLUSTER_NAME (0–2 nós)"

# --- Conclusão ---
echo ""
echo "============================================================"
echo " AMBIENTE PRONTO!"
echo "============================================================"
echo ""
echo " Próximos passos:"
echo "   1. Acesse: https://ml.azure.com"
echo "   2. Selecione o workspace: $WS_NAME"
echo "   3. Faça upload de data/heart.csv para o Blob Storage"
echo "   4. Abra o notebook e instale: pip install -r requirements-azure.txt"
echo ""
echo " Para encerrar e deletar todos os recursos:"
echo "   bash scripts/end-environment.sh"
echo "============================================================"
