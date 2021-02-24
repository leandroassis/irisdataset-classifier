# irisdataset-classifier

## Final work of Linear Algebra II

### Objetivos:
- Calcular os coeficientes de um sistema linear através dos Mínimos Quadrados e PLU+backsubstitution
- Calcular a Decomposição SVD
- Calcular a Decomposição Espectral
- Criar algoritmo de classificação com dois algoritmos distintos


## Importante:

### Separação de cada classe: 

Existem 3 possibilidades para separação das matrizes para cada classe:

1-)  Utiliza a função Separator(sem declarar altResponse e trainerMode) para separar o dataset em matrizes A 15x4 (ou 15x5) com os dados referentes à cada classe e b formada por 15 linhas de 1.
EX: A para classe irís-setosa       b todos os casos (nessa implementação)
    A = [[5.8 4.  1.2 0.2]          b = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
         [5.7 4.4 1.5 0.4]
         [5.4 3.9 1.3 0.4]
         [5.1 3.5 1.4 0.3]
         [5.7 3.8 1.7 0.3]
         [5.1 3.8 1.5 0.3]
         [5.4 3.4 1.7 0.2]
         [5.1 3.7 1.5 0.4]
         [4.6 3.6 1.  0.2]
         [5.1 3.3 1.7 0.5]
         [4.8 3.4 1.9 0.2]
         [5.  3.  1.6 0.2]
         [5.  3.4 1.6 0.4]
         [5.2 3.5 1.5 0.2]
         [5.2 3.4 1.4 0.2]] 

2-) Utiliza a função Separator (declarando altResponse = False e trainerMode = True) para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas. 
Ex: 
b para Iris-versicolor
    b = [[0]
         [0]
         [0]
         [0] 
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [1]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]
         [0]]
A para todos os casos
    A = [[5.8 4.  1.2 0.2]  
         [5.7 4.4 1.5 0.4]
         [5.4 3.9 1.3 0.4]
         [5.1 3.5 1.4 0.3]
         [5.7 3.8 1.7 0.3]
         [5.1 3.8 1.5 0.3]
         [5.4 3.4 1.7 0.2]
         [5.1 3.7 1.5 0.4]
         [4.6 3.6 1.  0.2]
         [5.1 3.3 1.7 0.5]
         [4.8 3.4 1.9 0.2]
         [5.  3.  1.6 0.2]
         [5.  3.4 1.6 0.4]
         [5.2 3.5 1.5 0.2]
         [5.2 3.4 1.4 0.2]
         [5.6 2.9 3.6 1.3]
         [6.7 3.1 4.4 1.4]
         [5.6 3.  4.5 1.5]
         [5.8 2.7 4.1 1. ]
         [6.2 2.2 4.5 1.5]
         [5.6 2.5 3.9 1.1]
         [5.9 3.2 4.8 1.8]
         [6.1 2.8 4.  1.3]
         [6.3 2.5 4.9 1.5]
         [6.1 2.8 4.7 1.2]
         [6.4 2.9 4.3 1.3]
         [6.6 3.  4.4 1.4]
         [6.8 2.8 4.8 1.4]
         [6.7 3.  5.  1.7]
         [6.  2.9 4.5 1.5]
         [5.8 2.8 5.1 2.4]
         [6.4 3.2 5.3 2.3]
         [6.5 3.  5.5 1.8]
         [7.7 3.8 6.7 2.2]
         [7.7 2.6 6.9 2.3]
         [6.  2.2 5.  1.5]
         [6.9 3.2 5.7 2.3]
         [5.6 2.8 4.9 2. ]
         [7.7 2.8 6.7 2. ]
         [6.3 2.7 4.9 1.8]
         [6.7 3.3 5.7 2.1]
         [7.2 3.2 6.  1.8]
         [6.2 2.8 4.8 1.8]
         [6.1 3.  4.9 1.8]
         [6.4 2.8 5.6 2.1]]
3-) Utiliza a função Separator(declarando trainerMode e altReponse = True) para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com -1 nas linhas referentes à classe Iris-setosa, 0 nas refernetes à iris-versicolor e 2 nas referentes à iris-virgínica (método utilizado apenas no classificador StepFunction). 

Escolhi usar a segunda alternativa para realizar todo o projeto pois foi a que obteve o melhor desempenho.

### Algoritmos de classificação:



## Funcionamento:

- Questão 1.1:
Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
Passo 2: Soluciona o sistema Ax = b utilizando o mínimos quadrados. 

- Questão 1.2:
Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
Passo 3: Utiliza as funções PLU e backsubstitution para resolver o sistema e obter os mesmos coeficientes acima.
Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 2:
Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
Passo 3: Calcula a decomposição espectral (atualmente utilizando a função do módulo numpy -novidade em breve rs-)
Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 3:
Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
Passo 3: Calcula a decomposição SVD (atualmente utilizando a função do módulo numpy) 
Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 4: