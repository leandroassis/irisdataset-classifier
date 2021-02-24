import numpy as np 
import pandas as pd

'''
To do Comentar e refatorar código
To do Classificador apontar casos de indecisão
To do Fazer Power Method
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

    def createB(self, datas, typeFlower="", trainerON=False):
        array = []
        if typeFlower != "":
            for linha in datas:
                if type(linha) == str:
                    if trainerON == True:
                        if linha == typeFlower:
                            aux = 1
                        else:
                            aux = 0
                        array.append([aux])
                    else:
                        if linha == typeFlower:
                            aux = 1
                            array.append([aux])
                        else:
                            continue
        else: #Parte usada unicamente para treinar o algoritmo de classificação Step Function
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

    def Separator(self, dataSet, flowerType="", bias=None, altResponse=None, trainerMode = False):
        data = pd.read_csv(dataSet)
        Species = data["Species"]
        sepalLength = data["SepalLengthCm"]
        sepalWidth = data["SepalWidthCm"]
        petalLength = data["PetalLengthCm"]
        petalWidth = data["PetalWidthCm"]
        
        if altResponse == True:
            self.b = self.createB(Species,trainerON=trainerMode)
            self.A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth, bias)
        elif altResponse == False:
            self.b = self.createB(Species, flowerType,trainerON=trainerMode)
            self.A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth, bias)
        else:
            self.b = self.createB(Species, flowerType,trainerON=trainerMode)
            sLength, sWidth, pLength, pWidth = [],[],[],[]
            for line in range(len(Species)):
                if Species[line] == flowerType:
                    sLength.append(sepalLength[line])
                    sWidth.append(sepalWidth[line])
                    pLength.append(petalLength[line])
                    pWidth.append(petalWidth[line])
                else:
                    continue
            self.A = self.createA(sLength, sWidth, pLength, pWidth, bias)        

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
        self.Separator(trainDataSet, typeFlower, bias, trainerMode=True, altResponse=False)
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
        self.Separator(trainDataSet, bias=bias, altResponse=True, trainerMode=True)
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

def run():
    objct = Iris_Classifier()

    dataSet = "dados_08.csv"
    data = pd.read_csv(dataSet)
    bias = None
    specie = ""

    print("Trabalho Final de Álgebra Linear")
    print("Grupo 8 - Leandro Assis, Paulo Victor Lima, Pedro Alonso, Victor Nunes")

    #Questão 1
    print("\nQuestão 1.1: Coeficientes para cada classe usando Mínimos Quadrados\n")
    for flower in data["Species"]:
        if specie == flower:
            continue
        else:
            objct.Separator(dataSet, flower,bias, trainerMode=True, altResponse=False) #para incluir o bias basta modificar o valor de bias na declaração no top dessa função
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
                objct.Separator(dataSet, flower,bias, trainerMode=True, altResponse=False) #Separando cada uma das classes
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
                objct.Separator(dataSet, flower, bias, trainerMode=True, altResponse=False) #para incluir o bias basta modificar o valor de bias na declaração no top dessa função
                A,b = objct.NormalEquation(objct.A, objct.b) #discarta-se o B pois o interesse é apenas fazer a decomposição de A
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
                objct.Separator(dataSet, flower, bias, trainerMode=True, altResponse=False) #para incluir o bias basta modificar o valor de bias na declaração no top dessa função
                A,b = objct.NormalEquation(objct.A, objct.b) #discarta-se o B pois o interesse é apenas fazer a decomposição de A
                print("Decomposição SVD da matriz A da classe "+flower)
                U,s,V = np.linalg.svd(A)
                print("A matriz U é ")
                print(U)
                print(s)
                print(V)
                print("\n")
                specie = flower
        if bias != 1:
            print("\n--- Com bias ---:")
            bias = 1
    bias = None

    #Questão 4
    print("\nQuestão 4: Classificando as amostras\n")
    print("OBS: A primeira resposta é referente ao classificador Um contra Todos e a segunda ao Algoritmo com Step Function")
    print("A-)")
    objct.OneVsAllAlgorithm(5,2.3,3.3,1,bias=bias)
    objct.StepFunctionAlgorithm(5,2.3,3.3,1,bias=bias)
    print("\nB-)")
    objct.OneVsAllAlgorithm(4.6,3.2,1.4,0.2,bias=bias)
    objct.StepFunctionAlgorithm(4.6,3.2,1.4,0.2,bias=bias)
    print("\nC-)")
    objct.OneVsAllAlgorithm(5.0,3.3,1.4,0.2,bias=bias)
    objct.StepFunctionAlgorithm(5.0,3.3,1.4,0.2,bias=bias)
    print("\nD-)")
    objct.OneVsAllAlgorithm(6.1,3.0,4.6,1.4,bias=bias)
    objct.StepFunctionAlgorithm(6.1,3.0,4.6,1.4,bias=bias)
    print("\nE-)")
    objct.OneVsAllAlgorithm(5.9,3.0,5.1,1.8,bias=1)
    objct.StepFunctionAlgorithm(5.9,3.0,5.1,1.8,bias=bias)

    print("\nTestes Extra:")
    print("Classificador um contra todos")
    objct.OneVsAllAlgorithm(dataSet="dados_08.csv", bias=bias) #Testando todo o dataset para medir a eficiência do algoritmo
    print("\nClassificador StepFunction")
    objct.StepFunctionAlgorithm(dataSet="dados_08.csv", bias=bias)

    '''
    As diferenças de acurácia entre os dois algoritmos se dá pela forma de implementação.
    O algoritmo um contra todos dá como classificação a saída que obtiver maior "probabilidade", já o step function apenas realiza uma função degrau para levar o valor para\
        um dos três valores -1,1,2.
    Por conta dessas características há uma diferença na classificação, já que o UmcontraTodos pode ficar na dúvida entre qual classe escolher (essa situação seria em que os 3\
        classificadores apontam uma porcentagem baixa). Caso isso ocorra a maior "porcentagem" será retornada, mas não necessariamente essa é a decisão do algoritmo já que \
            no fundo há uma indecisão. Por outro lado o Step Function apenas leva o valor de saída do classificador para o indicador mais próximo - entre -1,1 e 2 -, funcionando \
                similarmente a um "chute" bem dado.
    O conceito de probabilidade descrito acima é o entendimento de cada classificador sobre a posição do ponto passado, ou seja, se está dentro da região característica daquele\
        classificador ou não.
    '''

if __name__ == "__main__":
    run()