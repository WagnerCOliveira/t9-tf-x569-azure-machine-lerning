# 1. Configuração de Variáveis (Nomes personalizados para o Trabalho Final)
RG_NAME="rg-projeto-final-ml"
LOCATION="eastus"
WS_NAME="mlw-projeto-final"
CI_NAME="ci-notebooks-trabalho"
CLUSTER_NAME="cluster-treino-trabalho"

echo "Instalando/Atualizando a extensão Azure ML (necessário para os comandos 'az ml')..."
az extension add -n ml -y

echo "--------------------------------------------------------"
echo "Criando Grupo de Recursos: $RG_NAME..."
az group create --name $RG_NAME --location $LOCATION

echo "Criando o Workspace: $WS_NAME (isso demora cerca de 3 a 5 min)..."
az ml workspace create --name $WS_NAME -g $RG_NAME -l $LOCATION

echo "Criando Instância de Computação (para rodar seus Notebooks)..."
az ml compute create --name $CI_NAME \
    --size STANDARD_DS11_V2 \
    --type ComputeInstance \
    -w $WS_NAME -g $RG_NAME

echo "Criando Cluster de Computação (para treinos pesados)..."
# Configurado com min-instances 0 para não gastar crédito parado
az ml compute create --name $CLUSTER_NAME \
    --size STANDARD_DS11_V2 \
    --min-instances 0 \
    --max-instances 2 \
    --type AmlCompute \
    -w $WS_NAME -g $RG_NAME

echo "--------------------------------------------------------"
echo "AMBIENTE PRONTO!"
echo "Acesse: https://ml.azure.com e selecione o workspace $WS_NAME"