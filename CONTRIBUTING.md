# Guia de Contribuição

Obrigado pelo interesse em contribuir com o projeto! Este documento descreve as diretrizes para contribuições.

---

## Como Contribuir

### 1. Reportar Issues

Para reportar bugs ou sugerir melhorias:
1. Verifique se o issue já existe na lista de issues do repositório
2. Abra um novo issue com:
   - Título claro e descritivo
   - Descrição detalhada do problema ou sugestão
   - Passos para reproduzir (para bugs)
   - Ambiente: OS, versão do Python, versão das bibliotecas

### 2. Submeter Pull Requests

```bash
# 1. Fork o repositório e clone localmente
git clone <url-do-seu-fork>
cd projeto

# 2. Crie uma branch descritiva
git checkout -b feat/novo-modelo
# ou
git checkout -b fix/corrige-preprocessing

# 3. Faça as alterações e commite
git add <arquivos-modificados>
git commit -m "feat: adiciona modelo Random Forest com SHAP"

# 4. Push e abra o PR
git push origin feat/novo-modelo
```

---

## Padrões de Código

### Python

- Seguir [PEP 8](https://peps.python.org/pep-0008/) para estilo de código
- Usar type hints em funções quando aplicável
- Docstrings em funções públicas (formato NumPy ou Google)
- Linhas com no máximo 100 caracteres

```python
# Bom
def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> object:
    """Treina um classificador com os parâmetros fornecidos."""
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model

# Ruim
def train(x,y,p):
    m = LGBMClassifier(**p)
    m.fit(x,y)
    return m
```

### Notebooks

- Reiniciar o kernel e executar todas as células antes de commitar
- Limpar outputs de células com dados sensíveis
- Manter células de Markdown explicativas entre blocos de código
- Usar `# %%` para separar seções lógicas

### Commits

Seguir o padrão [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: adiciona novo modelo Gradient Boosting
fix: corrige imputação de valores ausentes em 'ca'
docs: atualiza DATA_DICTIONARY com novas features
refactor: extrai função de pré-processamento para módulo separado
test: adiciona testes de validação do pipeline
chore: atualiza versão do scikit-learn para 1.5.1
```

---

## Estrutura de Branches

| Branch | Propósito |
|--------|-----------|
| `main` | Código estável, revisado |
| `feat/*` | Novas funcionalidades |
| `fix/*` | Correção de bugs |
| `docs/*` | Atualizações de documentação |
| `refactor/*` | Refatorações sem mudança de comportamento |

---

## Checklist para Pull Requests

Antes de abrir um PR, verifique:

- [ ] O código segue os padrões de estilo (PEP 8)
- [ ] Notebooks foram executados do zero sem erros
- [ ] Nenhuma credencial ou dado sensível no código
- [ ] Documentação atualizada se necessário (README, docs/)
- [ ] Testes passam localmente (se aplicável)
- [ ] Branch atualizada com `main`

---

## Code of Conduct

Este projeto segue o [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Em resumo:

- Seja respeitoso e inclusivo nas interações
- Aceite críticas construtivas com abertura
- Foque no que é melhor para o projeto
- Sem discriminação de qualquer natureza

---

## Dúvidas?

Abra uma issue com a tag `question` ou entre em contato pelo e-mail do autor no [README.md](README.md).
