# irisdataset-classifier

## Final work of Linear Algebra II

### Execução:
Para obter as respostas de cada pergunta basta:
- Clonar o repositório (requer git instalado "sudo apt-get install git-all"):
```
    git clone https://github.com/leandroassis/irisdataset-classifier.git
```
- Executar:
```
    pip3 install numpy && pip3 install pandas && clear && python3 grupo8.py
```
### Objetivos:
- Calcular os coeficientes de um sistema linear através dos Mínimos Quadrados e PLU+backsubstitution
- Calcular a Decomposição SVD
- Calcular a Decomposição Espectral
- Criar algoritmo de classificação com dois algoritmos distintos


## Importante:

### Separação de cada classe: 

Existem 3 possibilidades para separação das matrizes para cada classe:

1. Utiliza a função Separator(sem declarar altResponse e trainerMode) para separar o dataset em matrizes A 15x4 (ou 15x5) com os dados referentes à cada classe e b formada por 15 linhas de 1. (--> Essa forma, apesar de muito similar a seguinte, retorna coeficientes que não resultam em classificações estáveis)
```
EX: A para classe irís-setosa       b todos as classes (nessa implementação)
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
```
2. Utiliza a função Separator (declarando altResponse = False e trainerMode = True) para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas. 
``` 
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
A para todas as classes (nessa implementação e na posterior)
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
```
3. Utiliza a função Separator(declarando trainerMode e altReponse = True) para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com -1 nas linhas referentes à classe Iris-setosa, 1 nas refernetes à iris-versicolor e 2 nas referentes à iris-virgínica (método utilizado apenas no classificador StepFunction).
```
Ex:
b para todas as classes:
b = [[-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [-1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 1]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]
     [ 2]]
```

Escolhi usar a segunda alternativa para realizar todo o projeto pois foi a que obteve o melhor desempenho.

### Algoritmos de classificação:

Existem 2 algoritmos que fazem a classificação baseados nos valores passados para SLength, SWidth, PLength e PWidth. 

1. One Vs All:
Nesse algoritmo é utilizada a segunda forma de separação de dados descrita acima. A ideia é especializar três funções em detectar se a classe é ou não algo, ou seja, uma função especializada em Iris-setosa, outra em iris-versicolor e outra em iris-virginica. Cada função gera coeficientes diferentes (resolução do sistema linear Ax=b utilizando A e B descritos na forma 2 do tópico anterior), faz a operação dos valores de entrada por eles e retorna a chance daquele conjunto de pontos se enquadrar ou não na reta característica da classe. A função classificadora que retornar a maior chance têm-se como a classe correta.

Note que o benefício desse algoritmo é perceber a incerteza da máquina sobre a classificação de alguns pontos. Essa incerteza é visível quando as 3 funções classificadoras retornam chances muito baixas.

2. Algoritmo com Step Function:
Nesse algoritmo é utilizada a última forma de separação de dados descrita acima. A ideia é gerar os coefientes (resolução do sistema Ax = b usando A e b do tópico acima) e passar todas as entradas por eles. A resposta passa por um função degrau, que verifica qual valor de classe a resposta mais se aproxima (os valores de classe são: -1->Setosa, 1->Versicolor 2->Virginica).

Esse algoritmo, diferentemente do anterior, não consegue expressar suas incertezas em relação a classificação da amostra, apesar disso ele obteve um desempenho de 97,78% utilizando 45 dados para treinamento e chegou a 100% utilizando 150 dados para treinamento.

## Funcionamento:

- Questão 1.1:
    - Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
    - Passo 2: Soluciona o sistema Ax = b utilizando o mínimos quadrados. 

- Questão 1.2:
    - Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
    - Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
    - Passo 3: Utiliza as funções PLU e backsubstitution para resolver o sistema e obter os mesmos coeficientes acima.
    - Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 2:
    - Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
    - Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
    - Passo 3: Calcula a decomposição espectral (atualmente utilizando a função do módulo numpy -novidade em breve rs-)
    - Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 3:
    - Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 45x4 (ou 45x5) com os dados referentes à cada classe e b formada por 45 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
    - Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
    - Passo 3: Calcula a decomposição SVD (atualmente utilizando a função do módulo numpy) 
    - Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 4:
    - Passo 1: Chama as funções de cada algoritmo de classificação, que internamente treinam (geram coeficientes) para serem utilizados na classificação.
    - Passo 2: Resolve o sistema Ax=b, sendo A a matriz com os dados de pétala e sépala passados.
    - Passo 3: Apresenta a classificação.

## Notas:

A separação das matrizes em todo a resolução dos problemas está sendo feita seguindo a forma 2 do tópico "Separação das classes". Por conta disso, como era de se esperar, os autovalores e autovetores, assim como os componentes da decomposição SVD são os mesmos para as 3 classes. Isto se dá pois todas as decomposições estão sendo feitas sobre a mesma matriz A (descrita no item 2 do tópico "Sepração das classes").
Mantive como padrão essa apresentação para uniformizar a forma de separação para resolução dos problemas 1,2,3 e a forma de separação para treino do classificador que responde a questão 4. Caso entendam que essa separação é insatisfatória pode-se troca-la por qualquer uma das 3 citadas no tópico "Separação das classes" (Entretanto somente trocando pelo método 1 resultaria em diferenças entre os componentes das decomposições para cada classe). É importante salientar que, realizando essa troca para responder as questões 1,2 e 3, os coeficientes obtidos na questão 1 (através do mínimos quadrados e PLU+backsubstitution) não seriam os mesmos que os usados pelos algoritmos de classificação - oneVsAll e StepFunciton - pois ambos usam abordagens diferentes (usam as separações descritas nos itens 2 e 3, respectivamente, do tópico "Separação das classes").

Ex pronto para troca da forma de separação(que vai gerar componentes diferentes para cada classe nas decomposições SVD e Espectral):

```
    Basta substituir o código das questões 1,2 e 3 no .py pelo código abaixo

    #Questão 1
    print("\nQuestão 1.1: Coeficientes para cada classe usando Mínimos Quadrados\n")
    for flower in data["Species"]:
        if specie == flower:
            continue
        else:
            objct.Separator(dataSet, flower,bias) #para incluir o bias basta modificar o valor de bias na declaração no top dessa função
            print("Coeficientes da classe "+flower+".")
            print(objct.leastSquares(objct.A, objct.b))
            print("\n")
            specie = flower

    print("\nQuestão 1.2: Coeficientes para cada classe usando PLU + backsubstitution\n")
    print("--- Sem bias ---:")
    for execution in range(2):
        for flower in data["Species"]:
            if specie == flower:
                continue
            else:
                objct.Separator(dataSet, flower,bias) #Separando cada uma das classes
                A,b = objct.NormalEquation(objct.A, objct.b) #Criando equação normal
                A, b = objct.PLU(A,b) #executa o PLU
                coefficients = objct.backSubstitution(A,b) #Faz o backsubstitution em cima das matrizes A e b resultantes do PLU
                print("Coeficientes da classe "+flower+".")
                print(coefficients)
                print("\n")
                specie = flower
        if bias != 1:
            print("\n--- Com bias ---:")
            bias = 1
    bias = None

    #Questão 2
    print("\nQuestão 2: Decomposição Espectral\n")
    print("--- Sem bias ---:")
    for execution in range(2):
        for flower in data["Species"]:
            if specie == flower:
                continue
            else:
                objct.Separator(dataSet, flower, bias) 
                A,b = objct.NormalEquation(objct.A, objct.b) 
                print("Decomposição Espectral da matriz A da classe "+flower)
                print("\nA antes da decomposição: \n")
                print(A)
                eigenvalues, eigenvectors = np.linalg.eig(A)
                print("\nOs autovalores são: (alocados em forma de vetor)\n")
                print(eigenvalues)
                print("\nOs autovetores são:\n")
                print(eigenvectors)
                eigenvalues = np.diag(eigenvalues)
                A = eigenvectors.dot(eigenvalues).dot(np.linalg.inv(eigenvectors)) #remonta A
                print("\nA = PDP^-1, P = autovetores, D = matriz diagonal dos autovalores.\n")
                print(A)
                print("\n")
                specie = flower
        if bias != 1:
            print("\n--- Com bias ---:")
            bias = 1
    bias = None

    #Questão 3
    print("\nQuestão 3: Decomposição SVD\n")
    print("--- Sem bias ---:")
    for execution in range(2):
        for flower in data["Species"]:
            if specie == flower:
                continue
            else:
                objct.Separator(dataSet, flower, bias) 
                A,b = objct.NormalEquation(objct.A, objct.b) 
                print("Decomposição SVD da matriz da classe "+flower)
                U,s,V = np.linalg.svd(A)
                print("\nA matriz U:")
                print(U)
                print("\nOs valores singulares são:")
                print(s)
                print("\nA matriz V:")
                print(V)
                print("\n")
                specie = flower
        if bias != 1:
            print("\n--- Com bias ---:")
            bias = 1
    bias = 1
```