# Homer x Bart: classificação de imagens com transfer learning (VGG16)

Checkpoint 05 · Redes Neurais Artificiais, Deep Learning e Algoritmos Genéticos · FIAP 2026

**Aluno:** Bruno Henrique Ferreira Ambrosio · RM 571218

## O problema

Classificar a imagem de um personagem dos Simpsons como **Homer** ou **Bart**. É uma classificação binária com um dataset pequeno:

| Conjunto | Bart | Homer | Total |
|---|---|---|---|
| Treino | 118 | 78 | 196 |
| Teste | 42 | 31 | 73 |

Com só 196 imagens de treino, uma CNN treinada do zero decoraria os exemplos. Por isso o modelo usa **transfer learning** com a VGG16, pré-treinada no ImageNet.

## O que foi feito

1. **Pré-processamento** com o `preprocess_input` da VGG16, sem `rescale`, e data augmentation só no treino.
2. **Fase 1 (feature extraction):** a base da VGG16 fica congelada e só a nova cabeça densa é treinada.
3. **Fase 2 (fine-tuning):** o último bloco convolucional é liberado, com learning rate bem menor.
4. **Avaliação** no conjunto de teste, com relatório de classificação.
5. **Predição em imagem da internet** e **Grad-CAM**, que mostra em que parte da imagem a rede se baseou para decidir.

O ponto de partida foi o notebook da aula de transfer learning (Gatos vs Cachorros). O dataset, as classes, os hiperparâmetros e as análises foram adaptados para este problema.

## Resultados

| Configuração | Acurácia no teste |
|---|---|
| Fase 1 · feature extraction | 83,56% |
| Fase 2 · fine-tuning | **87,67%** |

## Como abrir

O notebook [`Homer_x_Bart_VGG16.ipynb`](Homer_x_Bart_VGG16.ipynb) já está executado e pode ser lido direto no GitHub.

Para rodar de novo, abra no Google Colab com GPU e envie o arquivo `dataset_personagens.zip` quando o notebook pedir.

**Tecnologias:** Python, TensorFlow/Keras (VGG16), NumPy e matplotlib.
