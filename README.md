# **irisdataset-classifier** - *Linear Algebra II Coursework*
## Made by: Leandro Assis, Paulo Victor Lima, Pedro Alonso e Victor Nunes.
## Date: 05/03/21

# Introduction

# Usage

To get the answer of each question you should just:
    1. Clone the repository (requires git installed "sudo apt-get install git-all"):
```
    git clone https://github.com/leandroassis/irisdataset-classifier.git
```
    2. Execute the follow command line into irisdataset-classifier folder:
```
    pip3 install numpy && pip3 install pandas && clear && python3 grupo8.py
```

# Goals:

* Calculate the coefficients of a linear system through Least Squares Method and PLU+backsubstitution.
* Calculate the Singular Values Decomposition (SVD)
* Calculate the Espectral Descomposition
* Create an IA classifier algorithm with two different algorithms (One vs All and Step function)


# Important:

## Each flower class separation: 

There are three possibilities to detach the each class matrices:

    1. Using the Separator functions (without declaring altResponse and trainerMode) to detach the dataset into matrices A 15x4 (or 15x5 if using bias) with each flower class data, e matrices b 15x1. (OBS: This possibility, although very similar to the following, return coefficients that result in stable classifications)
```
EX: A to iris setosa class          b to all classes
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
    2. Using the Separator functions (declaring altResponse = False and trainerMode = True) to detach the dataset into matrices A 45x4 (or 45x5 if using bias) with all data of each class mixed, and b made of 45 lines with 1 in the lines referents to the class being analyzed and 0 in the others else. 
``` 
Ex:
b to Iris-versicolor
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
A to all classes
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
    3. Using the Separator function (declaring trainderMode and altResponse = True) to detach the dataset into matrices A 45x4 (or 45x5 if using bias) with data of each class and b made of 45 lines with -1 on lines referents to Iris-Setosa class, 1 on lines referents to Iris-Versicolor and 2 on lines referents to Iris-Virginica (method used only in StepFunction Classifier Algorithm).
```
Ex:
b to all classes:
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

I choosed use the second possibilty to complete this coursework because it had the better results.

## Classifier Algorithms:

There are two classifier algorithms based on values passed to SLength, SWidth, PLength and PWidth. 

1. One Vs All:
In this algorithm is used the second possibility described above. The idea is to master three functions (one for each class) to calculate the probability of the flower with SLength, SWidth, PLength, and PWidth measures are from the respective class. In other words, there is one function specialized in Iris-Setosa, another one in Iris-Versicolor, and one last in Iris-Virginica. The function to return the highest probability is the flower classification.

Note that the benefit of this algorithm is shown the uncertainty in some classifications. That uncertainty is visible when the three function returns very low probabilities.

2. Step Function Algorithm:
In this algorithm is used the last possibility described above. The idea is generate the coefficients (results of solving de system Ax = b using A and b as described in the previous topic) and pass all the inputs for it. This operation result goes through a step function that check what predefined class value (-1 to Setosas, 1 to Versicolor and 2 to Virginica) it's closer.

This algorithm, in a different way from the previous, can't show uncertainty, although it had a performance of 97,78% using 45 dataset lenght to training and 100% using 130 dataset lenght to training.

## Operation:
#### In portuguese (ptBR) because i'm too tired to translate it. Sorry :p

- Questão 1.1:
    - Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 15x4 (ou 15x5) com os dados referentes à cada classe e b formada por 15 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
    - Passo 2: Soluciona o sistema Ax = b utilizando o mínimos quadrados. 

- Questão 1.2:
    - Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 15x4 (ou 15x5) com os dados referentes à cada classe e b formada por 15 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
    - Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
    - Passo 3: Utiliza as funções PLU e backsubstitution para resolver o sistema e obter os mesmos coeficientes acima.
    - Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 2:
    - Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 15x4 (ou 15x5) com os dados referentes à cada classe e b formada por 15 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
    - Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
    - Passo 3: Calcula a decomposição espectral (atualmente utilizando a função do módulo numpy -novidade em breve rs-)
    - Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 3:
    - Passo 1: Utiliza a função Separator para separar o dataset em matrizes A 15x4 (ou 15x5) com os dados referentes à cada classe e b formada por 15 linhas com 1 nas linhas referentes a classe em questão e 0 nas outras linhas.
    - Passo 2: Calcula a equação normal do sistema (A^T.Ax = A^T.b), ou seja A = A^T.A e b = A^T.b.
    - Passo 3: Calcula a decomposição SVD (atualmente utilizando a função do módulo numpy) 
    - Passo 4: Repete os passos acima para cada classe com e sem bias(termo independente).

- Questão 4:
    - Passo 1: Chama as funções de cada algoritmo de classificação, que internamente treinam (geram coeficientes) para serem utilizados na classificação.
    - Passo 2: Resolve o sistema Ax=b, sendo A a matriz com os dados de pétala e sépala passados.
    - Passo 3: Apresenta a classificação.

