# Definindo o nome do grupo que criamos anteriormente
RG_NAME="rg-projeto-final-ml"

echo "--------------------------------------------------------"
echo "INICIANDO A DELEÇÃO DO AMBIENTE: $RG_NAME"
echo "Isso removerá o Workspace, Computações e Dados associados."
echo "--------------------------------------------------------"

# O comando --no-wait permite que o terminal seja liberado enquanto o Azure trabalha no background
# O --yes pula a confirmação manual
az group delete --name $RG_NAME --yes --no-wait

echo "O comando de deleção foi enviado com sucesso!"
echo "O processo pode levar alguns minutos para ser concluído no Azure."
echo "Você pode acompanhar o status pelo Portal (portal.azure.com)."
echo "--------------------------------------------------------"