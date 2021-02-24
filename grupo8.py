import numpy as np 
import pandas as pd

'''
To do desvendar se na 1 é pra fazer seguindo metodologia OneVsAll ou não 
## metodologia OneVsAll = para cada classe com e sem bias fazer todo o dataset(A) ser igual a um b que possui valores 1 nas linhas referentes a classe e 0 nas outras (Em Uso)
## Não usar OneVsAll = para cada classe com e sem bias A tem apenas as linhas do dataset em que a classe é igual a classe em questão e b é uma matriz com -1 nas linhas referentes a Setosa \
1 nas linhas referentes a Versicolor e 2 nas referentes a Virginica

To do Comentar e refatorar código
To do Fazer apresentação das questões
To do Criar guia
To do Classificador apontar casos de indecisão
To do Decomposição SVD
To do Decomposição Espectral
'''

class Iris_Classifier():
    def highest(self, a,b,c):
        highestValue = a
        if b > highestValue:
            highestValue = b
            if c > highestValue:
                highestValue = c
        elif c > highestValue:
            highestValue = c
        return highestValue

    def createB(self, datas, typeFlower=""):
        array = []
        if typeFlower != "":
            for linha in datas:
                if type(linha) == str:
                    if linha == typeFlower:
                        aux = 1
                    else:
                        aux = 0
                    array.append([aux])
        else:
            for linha in datas:
                if type(linha) == str:
                    if linha == "Iris-setosa":
                        aux = -1
                    elif linha == "Iris-versicolor":
                        aux = 1
                    else:
                        aux = 2
                    array.append([aux])
        return np.array(array)

    def createA(self, sLength, sWidth, pLength, pWidth, bias = None):
        matrix = []
        for index in range(len(sLength)):
            if bias != None:
                matrix.append([sLength[index], sWidth[index], pLength[index], pWidth[index], bias])
            else:
                matrix.append([sLength[index], sWidth[index], pLength[index], pWidth[index]])
        return np.array(matrix)
    
    def leastSquares(self, A, b):
        x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
        return x

    def NormalEquation(self, A, b):
        A1 = A.transpose().dot(A)
        b = A.transpose().dot(b)
        return A1,b

    def Acurrancier(self, index, response,Species, hits=0, tests=0):
        if Species[index] == response:
            hits+=1
            tests+=1
        else:
            tests+=1
        return hits,tests

    def Separator(self, dataSet, flowerType="", bias=None, altResponse=None):
        data = pd.read_csv(dataSet)
        Species = data["Species"]
        sepalLength = data["SepalLengthCm"]
        sepalWidth = data["SepalWidthCm"]
        petalLength = data["PetalLengthCm"]
        petalWidth = data["PetalWidthCm"]
        
        if altResponse != None:
            self.b = self.createB(Species)
        else:
            self.b = self.createB(Species, flowerType)
        self.A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth, bias)
        
    def PLU(self, A, b, bias = False):  
        n = np.shape(A)[0]
        
        for column in range(len(A[0])):
            L = np.eye(n)
            for line in range(len(A)):
                if line == column:
                    item_diag = A[line][column]
                    if item_diag == 0:
                        for permutationLine in range(len(A)):
                            if permutationLine <= column -1:
                                continue
                            elif A[permutationLine][column] > item_diag:
                                item_diag = A[permutationLine][column]
                                pos_highest = permutationLine
                            else:
                                continue
                        # permutação da matriz A
                        aux = np.copy(A[line])
                        A[line] = A[pos_highest]
                        A[pos_highest] = aux    
                        #geração da P
                        aux = b[line]
                        b[line] = b[pos_highest]
                        b[pos_highest] = aux
                elif line <= column-1:
                    continue
                else:
                    L[line][column] = -1*(A[line][column]/item_diag)
            A = L.dot(A)
            b = L.dot(b)

        #algoritmo de organização de A
        for lines in range(len(A)):
            for columns in range(len(A[lines])):
                if abs(A[lines][columns])*1000 < 1:
                    A[lines][columns] = 0
                else:
                    continue
        return A,b

    def backSubstitution(self, A, b):
        coefficients = []
        if len(A[0]) == 4:
            x4 = b[3]/A[3][3] 
            x3 = (b[2] - A[2][3]*x4)/A[2][2]
            x2 = (b[1] - A[1][3]*x4 - A[1][2]*x3)/A[1][1]
            x1 = (b[0] - A[0][1]*x2 - A[0][2]*x3 - A[0][3]*x4)/A[0][0]
            coefficients += x1,x2,x3,x4
        else:
            x5 = b[4]/A[4][4]
            x4 = (b[3] - A[3][4]*x5)/A[3][3]
            x3 = (b[2] - A[2][4]*x5 - A[2][3]*x4)/A[2][2]
            x2 = (b[1] - A[1][4]*x5 - A[1][3]*x4 - A[1][2]*x3)/A[2][2]
            x1 = (b[0] - A[0][4]*x5 - A[0][3]*x4 - A[0][2]*x3 - A[0][1]*x2)/A[0][0]
            coefficients += x1,x2,x3,x4,x5
        return np.array(coefficients)

    def trainAlgorithm(self, trainDataSet, bias=None, typeFlower=""):
        self.Separator(trainDataSet, typeFlower, bias)
        '''
        # Utilizar o PLU + backsubstitution para resolver essa operação resultaria em uma perda de precisão devido aos arredondamentos durante o método
        A,b = self.NormalEquation(self.A,self.b)
        A, b = self.PLU(A,b)   
        self.coefficients = self.backSubstitution(A,b)
        '''
        self.coefficients = self.leastSquares(self.A, self.b)
    
    def OneVsAllAlgorithm(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet=0, bias = None):
        self.trainAlgorithm("dados_08.csv", bias, typeFlower="Iris-setosa")
        SetosaClassifiers = self.coefficients

        self.trainAlgorithm("dados_08.csv", bias, typeFlower="Iris-versicolor")
        VersicolorClassifiers = self.coefficients

        self.trainAlgorithm("dados_08.csv", bias, typeFlower="Iris-virginica")
        VirginicaClassifiers = self.coefficients

        if dataSet == 0: 
            if bias != None:
                isSetosa = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(SetosaClassifiers)
                isVersicolor = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(VersicolorClassifiers)
                isVirginica = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(VirginicaClassifiers)
            else:
                isSetosa = np.array([sLength, sWidth, pLength, pWidth]).dot(SetosaClassifiers)
                isVersicolor = np.array([sLength, sWidth, pLength, pWidth]).dot(VersicolorClassifiers)
                isVirginica = np.array([sLength, sWidth, pLength, pWidth]).dot(VirginicaClassifiers)

            highestCoef = self.highest(isSetosa, isVersicolor, isVirginica)
            if isSetosa == highestCoef:
                response = "Iris-setosa"
            elif isVirginica == highestCoef:
                response = "Iris-virginica"
            elif isVersicolor == highestCoef:
                response = "Iris-versicolor"
            print("A flor é uma "+response)
        else:
            data = pd.read_csv(dataSet)
            sepalLength = data["SepalLengthCm"]
            sepalWidth = data["SepalWidthCm"]
            petalLength = data["PetalLengthCm"]
            petalWidth = data["PetalWidthCm"]
            Species = data["Species"]
        
            aux1, aux2 = 0, 0
            for item in range(len(sepalLength)):
                if bias != None:
                    isSetosa = np.array([sepalLength[item], sepalWidth[item], petalLength[item], petalWidth[item], bias]).dot(SetosaClassifiers)
                    isVersicolor = np.array([sepalLength[item], sepalWidth[item], petalLength[item], petalWidth[item], bias]).dot(VersicolorClassifiers)
                    isVirginica = np.array([sepalLength[item], sepalWidth[item], petalLength[item], petalWidth[item], bias]).dot(VirginicaClassifiers)
                else:
                    isSetosa = np.array([sepalLength[item], sepalWidth[item], petalLength[item], petalWidth[item]]).dot(SetosaClassifiers)
                    isVersicolor = np.array([sepalLength[item], sepalWidth[item], petalLength[item], petalWidth[item]]).dot(VersicolorClassifiers)
                    isVirginica = np.array([sepalLength[item], sepalWidth[item], petalLength[item], petalWidth[item]]).dot(VirginicaClassifiers)

                highestCoef = self.highest(isSetosa, isVersicolor, isVirginica)
                if isSetosa == highestCoef:
                    response = "Iris-setosa"
                elif isVirginica == highestCoef:
                    response = "Iris-virginica"
                elif isVersicolor == highestCoef:
                    response = "Iris-versicolor"
                aux1, aux2 = self.Acurrancier(item, response, Species, aux1, aux2)
            print("Acurácia do algoritmo é de "+str(round(aux1/aux2*100,2))+"%")
            return aux1/aux2

    def trainStepAlgorithm(self, trainDataSet, bias):
        self.Separator(trainDataSet, bias=bias, altResponse=True)
        self.coefficients = self.leastSquares(self.A, self.b) #novamente utilizando mínimos quadrados para não perder eficiência
   
    def StepFunctionAlgorithm(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet=0, bias=None): #Função classificadora utilizando step Function
        #self.trainStepAlgorithm("iris2.csv", bias) #treinando com 150 dados para obter 100% de acurácia
        self.trainStepAlgorithm("dados_08.csv", bias) #treinando com 45 dados para obter 97.8% de acurácia
        if dataSet == 0: 
            if bias != None:
                classification = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(self.coefficients)[0]
            else:
                classification = np.array([sLength, sWidth, pLength, pWidth]).dot(self.coefficients)[0]

            if round(classification) <= -1:
                response = "Iris-setosa"
            elif round(classification) ==1:
                response = "Iris-virginica"
            else:
                response = "Iris-versicolor"
            print("A flor é uma "+response)
        else:
            data = pd.read_csv(dataSet)
            sepalLength = data["SepalLengthCm"]
            sepalWidth = data["SepalWidthCm"]
            petalLength = data["PetalLengthCm"]
            petalWidth = data["PetalWidthCm"]
            Species = data["Species"]

            aux1, aux2 = 0, 0
            for item in range(len(sepalLength)):
                if bias != None:
                    classification = np.array([sepalLength[item], sepalWidth[item], petalLength[item], petalWidth[item], bias]).dot(self.coefficients)
                else:
                    classification = np.array([sepalLength[item], sepalWidth[item], petalLength[item], petalWidth[item]]).dot(self.coefficients)
                if round(classification[0]) <= -1:
                    response = "Iris-setosa"
                elif round(classification[0]) ==1:
                    response = "Iris-versicolor"
                else:
                    response = "Iris-virginica"
                aux1, aux2 = self.Acurrancier(item, response, Species, aux1, aux2)
            print("Acurácia do algoritmo é de "+str(round(aux1/aux2*100,2))+"%")
            return aux1/aux2

    def powerMethod(self, A):
        error = 1e-8
        v = 0
        lamb = 0

        eigvectors = []
        eigvalues = []
        x = []
        
        #criação do vetor x 
        for line in range(np.shape(A)[0]):
            x.append([1])
        save_x = x
        x = np.array(save_x)
        
        
        x = A.dot(x)
        x = x/np.linalg.norm(x)

        for index in range(np.shape(A)[0]):
            x = save_x
            while True:
                x = A.dot(x)
                x_norm = np.linalg.norm(x)
                x = x/ x_norm
                if (abs(lamb - x_norm) <= error):
                    break
                else:
                    lamb = x_norm
            #removendo erros de aproximação do computador
            for line in range(len(x)):
                for column in range(len(x[line])):
                    if x[line][column]*10000 < 1:
                        x[line][column] = 0
            eigvectors.append(x)
            eigvalues.append(lamb)
            #print(eigvectors)
            #print(eigvalues)
            v = x/np.linalg.norm(x)
            A = A - lamb*v*v.transpose()

        return np.array(eigvalues), eigvectors

    def SpectralDecomposition(self, dataSet, bias= None, typeFlower=""):
        if bias:
            self.Separator(dataSet, typeFlower, bias)
            #print(self.powerMethod(self.A))
        else:
            self.Separator(dataSet, flowerType=typeFlower)
            A,b = self.NormalEquation(self.A, self.b)
            A = A.transpose().dot(A)
            values1, vectors1 = self.powerMethod(A)
            values, vectors = np.linalg.eig(A)
            print(values1)
            print(values)

def run():
    objct = Iris_Classifier()

    dataSet = "dados_08.csv"
    bias = None

    print("Trabalho Final de Álgebra Linear")
    print("Grupo 8 - Leandro Assis, Paulo Victor Lima, Pedro Alonso, Victor Nunes")

    #Questão 1
    print("\nQuestão 1.1: Coeficientes para cada classe usando Mínimos Quadrados\n")
    data = pd.read_csv(dataSet)
    specie = ""
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
                print("\n")
                specie = flower
        if bias != 1:
            print("\n--- Com bias ---:")
            bias = 1
    bias = None


    print("\nQuestão 4: Classificando as amostras\n")
    print("OBS: A primeira resposta é referente ao classificador Um contra Todos e a segunda ao Algoritmo com Step Function")
    print("A-)")
    objct.OneVsAllAlgorithm(5,2.3,3.3,1,bias=1)
    objct.StepFunctionAlgorithm(5,2.3,3.3,1,bias=1)
    print("\nB-)")
    objct.OneVsAllAlgorithm(4.6,3.2,1.4,0.2,bias=1)
    objct.StepFunctionAlgorithm(4.6,3.2,1.4,0.2,bias=1)
    print("\nC-)")
    objct.OneVsAllAlgorithm(5.0,3.3,1.4,0.2,bias=1)
    objct.StepFunctionAlgorithm(5.0,3.3,1.4,0.2,bias=1)
    print("\nD-)")
    objct.OneVsAllAlgorithm(6.1,3.0,4.6,1.4,bias=1)
    objct.StepFunctionAlgorithm(6.1,3.0,4.6,1.4,bias=1)
    print("\nE-)")
    objct.OneVsAllAlgorithm(5.9,3.0,5.1,1.8,bias=1)
    objct.StepFunctionAlgorithm(5.9,3.0,5.1,1.8,bias=1)

    print("\nTestes Extra:")
    print("Classificador um contra todos")
    objct.OneVsAllAlgorithm(dataSet="dados_08.csv", bias=1) #Testando todo o dataset para medir a eficiência do algoritmo
    print("\nClassificador StepFunction")
    objct.StepFunctionAlgorithm(dataSet="dados_08.csv")
    '''
    As diferenças de acurácia entre os dois algoritmos se dá pela forma de implementação.
    O algoritmo um contra todos dá como classificação a saída que obtiver maior "probabilidade", já o step function apenas realiza uma função degrau para levar o valor para\
        um dos três valores -1,1,2.
    Por conta dessas características há uma diferença na classificação, já que o UmcontraTodos pode ficar na dúvida entre qual classe escolher (essa situação seria em que os 3\
        classificadores apontam uma porcentagem baixa). Caso isso ocorra a maior "porcentagem" será retornada, mas não necessariamente essa é a decisão do algoritmo já que \
            no fundo há uma indecisão. Por outro lado o Step Function apenas leva o valor de saída do classificador para o indicador mais próximo - entre -1,1 e 2 -, funcionando \
                similarmente a um "chute" bem dado.
    '''

if __name__ == "__main__":
    run()