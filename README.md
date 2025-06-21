# Sistema de Reconhecimento de Libras Melhorado

Este projeto implementa um sistema avançado de reconhecimento de Libras usando visão computacional e machine learning, com foco especial na **melhoria da confiança das predições**.

## 🚀 Novas Funcionalidades de Confiança

### 1. Sistema de Votação Múltipla
- **Buffer de Predições**: Armazena as últimas 5 predições para análise
- **Votação por Maioria**: Só aceita uma letra se ela aparecer em pelo menos 3 predições consecutivas
- **Confiança Média**: Calcula a confiança média das predições votadas

### 2. Filtros de Validação
- **Verificação de Sequência**: Evita repetições excessivas da mesma letra
- **Threshold Dinâmico**: Permite ajustar a sensibilidade em tempo real
- **Validação de Consistência**: Rejeita sequências impossíveis

### 3. Interface Melhorada
- **Feedback Visual**: Mostra a confiança atual na tela
- **Controles de Sensibilidade**: Teclas `+` e `-` para ajustar o threshold
- **Informações em Tempo Real**: Exibe configurações atuais e status

### 4. Análise Avançada
- **Validação de Modelo**: Script dedicado para análise de confiança
- **Relatórios Detalhados**: Gera gráficos e relatórios de performance
- **Recomendações**: Sugere melhorias baseadas na análise

## 📁 Estrutura do Projeto

```
reading_hands_libras/
├── main.py                 # Programa principal com melhorias de confiança
├── train_model.py          # Treinamento com data augmentation avançado
├── validate_model.py       # Validação e análise de confiança
├── HandTrackingModule.py   # Módulo de detecção de mãos
├── requirements.txt        # Dependências atualizadas
├── README.md              # Este arquivo
└── model_hands/           # Pasta com modelos e dados
    ├── libras_model_improved.h5    # Modelo treinado
    ├── scaler_params.npy           # Parâmetros de normalização
    ├── training_data_*.csv         # Dados de treinamento
    ├── confidence_analysis.png     # Gráficos de análise
    └── confidence_report.txt       # Relatório de confiança
```

## 🛠️ Instalação

1. **Clone o repositório**:
```bash
git clone <url-do-repositorio>
cd reading_hands_libras
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

## 🎯 Como Usar

### 1. Coleta de Dados
```bash
python main.py
```
- Pressione `c` para coletar dados de treinamento
- Digite a letra desejada (A-Z)
- Faça o gesto da letra na frente da câmera
- Pressione `c` novamente para parar

### 2. Treinamento do Modelo
```bash
python train_model.py
```
- Treina o modelo com data augmentation avançado
- Usa cross-validation para melhor generalização
- Gera gráficos de treinamento

### 3. Validação e Análise
```bash
python validate_model.py
```
- Analisa a confiança do modelo
- Gera relatórios detalhados
- Cria gráficos de performance

### 4. Reconhecimento em Tempo Real
```bash
python main.py
```
- Pressione `l` para carregar o modelo
- Mostre sua mão para a câmera
- Use `+` e `-` para ajustar a sensibilidade
- Pressione espaço para limpar a palavra

## 🎮 Controles

| Tecla | Função |
|-------|--------|
| `c` | Coletar dados / Parar coleta |
| `l` | Carregar modelo |
| `+` | Aumentar sensibilidade |
| `-` | Diminuir sensibilidade |
| `Espaço` | Limpar palavra atual |
| `q` | Sair |

## 📊 Melhorias de Confiança

### Sistema de Votação
- **Buffer Size**: 5 predições consecutivas
- **Mínimo de Votos**: 3 para aceitar uma letra
- **Confiança Média**: Calculada das predições votadas

### Thresholds Recomendados
- **Inicial**: 0.7 (70% de confiança)
- **Ajustável**: 0.3 a 0.95 via teclas `+` e `-`
- **Ótimo**: Determinado automaticamente pela validação

### Validações Implementadas
- ✅ Evita repetições excessivas (AAA, BBB)
- ✅ Rejeita sequências impossíveis
- ✅ Verifica consistência temporal
- ✅ Filtra predições de baixa confiança

## 📈 Análise de Performance

O script `validate_model.py` gera:

1. **Relatório de Confiança**: Análise detalhada por letra
2. **Gráficos de Performance**: Visualização da accuracy e confiança
3. **Recomendações**: Sugestões para melhorar o modelo
4. **Threshold Ótimo**: Valor recomendado para melhor balance

### Métricas Analisadas
- **Accuracy por Letra**: Precisão individual de cada letra
- **Confiança Média**: Confiança típica do modelo
- **Coverage**: Porcentagem de amostras aceitas
- **Distribuição**: Análise da distribuição de dados

## 🔧 Configurações Avançadas

### Parâmetros do Sistema de Votação
```python
prediction_buffer = deque(maxlen=5)  # Buffer das últimas 5 predições
confidence_threshold = 0.7           # Threshold inicial
min_votes = 3                        # Mínimo de votos
letter_stability_time = 0.8          # Tempo de estabilização
```

### Data Augmentation
- **Ruído Gaussiano**: Variação de 1-3%
- **Escala**: ±5% de variação
- **Rotação**: ±5 graus
- **Amostras**: 5x aumento dos dados originais

## 🎯 Dicas para Melhor Confiança

1. **Coleta de Dados**:
   - Colete pelo menos 50 amostras por letra
   - Varie a posição e orientação da mão
   - Use diferentes condições de iluminação

2. **Uso do Sistema**:
   - Mantenha a mão estável por pelo menos 0.8 segundos
   - Ajuste a sensibilidade conforme necessário
   - Use boa iluminação e fundo neutro

3. **Otimização**:
   - Execute `validate_model.py` regularmente
   - Monitore o relatório de confiança
   - Colete mais dados para letras com baixa accuracy

## 🐛 Solução de Problemas

### Baixa Confiança
1. Colete mais dados de treinamento
2. Ajuste o threshold com `+` e `-`
3. Verifique a iluminação e posição da mão
4. Execute `validate_model.py` para análise

### Falsos Positivos
1. Aumente o threshold de confiança
2. Colete dados mais variados
3. Ajuste o sistema de votação
4. Verifique a qualidade dos dados de treinamento

### Modelo Não Carrega
1. Execute `train_model.py` primeiro
2. Verifique se os arquivos estão na pasta `model_hands/`
3. Confirme que as dependências estão instaladas

## 📝 Logs e Relatórios

O sistema gera automaticamente:
- **Logs de Treinamento**: Progresso e métricas
- **Relatório de Confiança**: Análise detalhada
- **Gráficos de Performance**: Visualizações
- **Configurações Salvas**: Parâmetros otimizados

## 🤝 Contribuição

Para contribuir com melhorias:
1. Teste o sistema com diferentes condições
2. Colete dados adicionais para letras problemáticas
3. Experimente diferentes configurações
4. Compartilhe relatórios de validação

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

---

**Desenvolvido com foco na acessibilidade e precisão do reconhecimento de Libras.**
