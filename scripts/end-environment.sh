#!/bin/bash
# =============================================================================
# Script: end-environment.sh
# Descrição: Remove todos os recursos Azure do projeto para evitar cobranças.
#            Deleta o Resource Group completo: Workspace, Compute, Blob Storage
#            e todos os recursos provisionados automaticamente pelo Workspace.
# Autor: Wagner Costa Oliveira
# Data: 2025-10-15
# Uso: bash scripts/end-environment.sh
# AVISO: Operação IRREVERSÍVEL. Modelos registrados e dados no Blob serão perdidos.
#        Faça backup local de artefatos importantes antes de executar.
# =============================================================================

# --- Configuração ---
RG_NAME="rg-projeto-final-ml"

echo "============================================================"
echo " LIMPEZA DO AMBIENTE AZURE ML"
echo "============================================================"
echo ""
echo " ATENÇÃO: Esta operação removerá PERMANENTEMENTE:"
echo "   - Resource Group : $RG_NAME"
echo "   - Workspace Azure ML e todos os recursos associados"
echo "   - Compute Instance e Compute Cluster"
echo "   - Datasets e modelos armazenados no Blob Storage"
echo ""
echo " Esta ação NÃO pode ser desfeita."
echo "============================================================"
echo ""

# --- Confirmar intenção do usuário ---
read -rp " Digite 'deletar' para confirmar a remoção: " CONFIRMACAO

if [ "$CONFIRMACAO" != "deletar" ]; then
  echo ""
  echo " Operação cancelada. Nenhum recurso foi removido."
  exit 0
fi

echo ""
echo "[1/1] Enviando comando de deleção do Resource Group: $RG_NAME..."
echo "      Usando --no-wait para liberar o terminal imediatamente."
echo "      O Azure processará a remoção em segundo plano (5–10 min)."

# --yes: pula confirmação interativa do Azure CLI
# --no-wait: libera o terminal enquanto o Azure trabalha em background
az group delete --name "$RG_NAME" --yes --no-wait

echo ""
echo "============================================================"
echo " COMANDO ENVIADO COM SUCESSO!"
echo "============================================================"
echo ""
echo " O processo de deleção está em andamento no Azure."
echo " Acompanhe o status em: https://portal.azure.com"
echo " (Seção: Resource Groups → $RG_NAME → Activity Log)"
echo ""
echo " Após conclusão, verifique que não há cobranças pendentes em:"
echo " https://portal.azure.com/#view/Microsoft_Azure_Billing"
echo "============================================================"
