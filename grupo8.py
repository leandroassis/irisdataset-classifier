import numpy as np 
import pandas as pd

class classifier():
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
        x = np.linalg.inv((A.transpose().dot(A))).dot(A.transpose()).dot(b)
        return x

    def Acurrancier(self, index, response,Species, hits=0, tests=0):
        if Species[index] == response:
            hits+=1
            tests+=1
        else:
            tests+=1
        return hits,tests

    def isSetosa(self, sLength, sWidth, pLength, pWidth, Species, bias=None):
        if bias != None:
            A = self.createA(sLength, sWidth, pLength, pWidth, bias)
        else:
            A = self.createA(sLength, sWidth, pLength, pWidth)
        b = self.createB(Species, "Iris-setosa")
        classifierIsSetosa = self.leastSquares(A,b)
        return classifierIsSetosa

    def isVersicolor(self, sLength, sWidth, pLength, pWidth, Species, bias=None):
        if bias != None:
            A = self.createA(sLength, sWidth, pLength, pWidth, bias)
        else:
            A = self.createA(sLength, sWidth, pLength, pWidth)
        b = self.createB(Species, "Iris-versicolor")
        classifierIsVersicolor = self.leastSquares(A,b)
        return classifierIsVersicolor
    
    def isVirginica(self, sLength, sWidth, pLength, pWidth, Species, bias=None):
        if bias != None:
            A = self.createA(sLength, sWidth, pLength, pWidth, bias)
        else:
            A = self.createA(sLength, sWidth, pLength, pWidth)
        b = self.createB(Species, "Iris-virginica")
        classifierIsVirginica = self.leastSquares(A,b)
        return classifierIsVirginica

    def trainAlgorithm(self, trainDataSet, alternativeAlgorithm = False, bias = None):
        train_data = pd.read_csv(trainDataSet)
        sLength = train_data["SepalLengthCm"]
        sWidth = train_data["SepalWidthCm"]
        pLength = train_data["PetalLengthCm"]
        pWidth = train_data["PetalWidthCm"]
        Species = train_data["Species"]

        if alternativeAlgorithm:
            if bias != None:
                self.setosaClassifiers = self.isSetosa(sLength, sWidth, pLength, pWidth, Species, bias)
                self.versicolorClassifiers = self.isVersicolor(sLength, sWidth, pLength, pWidth, Species, bias)
                self.virginicaClassifiers =self.isVirginica(sLength, sWidth, pLength, pWidth, Species, bias)
            else:
                self.setosaClassifiers = self.isSetosa(sLength, sWidth, pLength, pWidth, Species)
                self.versicolorClassifiers = self.isVersicolor(sLength, sWidth, pLength, pWidth, Species)
                self.virginicaClassifiers =self.isVirginica(sLength, sWidth, pLength, pWidth, Species)
        else:
            if bias == None:
                A = self.createA(sLength, sWidth, pLength, pWidth)
            else:
                A = self.createA(sLength, sWidth, pLength, pWidth, bias)
            b = self.createB(Species)
            self.coefficients = self.leastSquares(A,b)

    def OneVsAllAlgorithm(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet = 0, bias = None): 
        if dataSet == 0:
            if bias!= None:
                isSetosa = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(self.setosaClassifiers)[0]
                isVersicolor = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(self.versicolorClassifiers)[0]
                isVirginica = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(self.virginicaClassifiers)[0]
            else:
                isSetosa = np.array([sLength, sWidth, pLength, pWidth]).dot(self.setosaClassifiers)[0]
                isVersicolor = np.array([sLength, sWidth, pLength, pWidth]).dot(self.versicolorClassifiers)[0]
                isVirginica = np.array([sLength, sWidth, pLength, pWidth]).dot(self.virginicaClassifiers)[0]
        
            print("\nA classificação de cada classificador é: ")
            print(str(round(isSetosa*100,2))+"% de chances de ser setosa")
            print(str(round(isVersicolor*100,2))+"% de chances de ser versicolor")
            print(str(round(isVirginica*100,2))+"% de chances de ser virginica")

            highestCoef = self.highest(isSetosa, isVersicolor, isVirginica)
            if highestCoef == isSetosa:
                print("Logo, é uma Iris Setosa")
            elif highestCoef == isVirginica:
                print("Logo, é uma Iris Virginica")
            elif highestCoef == isVersicolor:
                print("Logo, é uma Iris Versicolor")
        else:
            data = pd.read_csv(dataSet)
            sLength = data["SepalLengthCm"]
            sWidth = data["SepalWidthCm"]
            pLength = data["PetalLengthCm"]
            pWidth = data["PetalWidthCm"]
            Species = data["Species"]

            aux1 = 0
            aux2 = 0
            a = 0
            for item in range(len(sLength)):
                if bias == None:
                    isSetosa = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item]]).dot(self.setosaClassifiers)[0]
                    isVersicolor = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item]]).dot(self.versicolorClassifiers)[0]
                    isVirginica = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item]]).dot(self.virginicaClassifiers)[0]
                    '''
                    print("\nA classificação de cada classificador é: ")
                    print(str(round(isSetosa*100,2))+"% de chances de ser setosa")
                    print(str(round(isVersicolor*100,2))+"% de chances de ser versicolor")
                    print(str(round(isVirginica*100,2))+"% de chances de ser virginica")
                    '''
                else:
                    isSetosa = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item], bias]).dot(self.setosaClassifiers)[0]
                    isVersicolor = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item], bias]).dot(self.versicolorClassifiers)[0]
                    isVirginica = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item], bias]).dot(self.virginicaClassifiers)[0]
                
                highestCoef = self.highest(isSetosa, isVirginica, isVersicolor)

                if isSetosa == highestCoef:
                    response = "Iris-setosa"
                elif isVirginica == highestCoef:
                    response = "Iris-virginica"
                elif isVersicolor == highestCoef:
                    response = "Iris-versicolor"
                aux1, aux2 = self.Acurrancier(item, response, Species, aux1, aux2)
            print("\nAcurácia do algoritmo é de "+str(aux1/aux2*100)+"%")
            return aux1/aux2
    
    def classifierAlgorithm(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet = 0, bias=None):
        if dataSet == 0:
            if bias != None:
                flowerClass = np.array([sLength, sWidth, pLength, pWidth,bias]).dot(self.coefficients)
            else:
                flowerClass = np.array([sLength, sWidth, pLength, pWidth]).dot(self.coefficients)

            if round(flowerClass[0]) <= 0:
                print("Iris-setosa")
            elif round(flowerClass[0]) == 1:
                print("Iris-versicolor")
            else:
                print("Iris-virginica")
        else:
            data = pd.read_csv(dataSet)
            sepalLength = data["SepalLengthCm"]
            sepalWidth = data["SepalWidthCm"]
            petalLength = data["PetalLengthCm"]
            petalWidth = data["PetalWidthCm"]
            Species = data["Species"]

            if bias!= None:
                flowerClass = self.createA(sepalLength, sepalWidth, petalLength, petalWidth, bias).dot(self.coefficients)
            else: 
                flowerClass = self.createA(sepalLength, sepalWidth, petalLength, petalWidth).dot(self.coefficients)
            aux1 = 0
            aux2 = 0
            for item in range(len(flowerClass)):
                print(flowerClass[item])
                if round(flowerClass[item][0]) <= -1:
                    response = "Iris-setosa"
                elif round(flowerClass[item][0]) == 1:
                    response = "Iris-versicolor"
                else:
                    response = "Iris-virginica"
                aux1, aux2 = self.Acurrancier(item, response, Species, aux1, aux2)
            print("Acurácia do algoritmo é de "+str(aux1/aux2*100)+"%")
            return aux1/aux2
         
    def Separator(self, dataSet, flowerType, bias=None):
        data = pd.read_csv(dataSet)
        Species = data["Species"]
        sepalLength = data["SepalLengthCm"]
        sepalWidth = data["SepalWidthCm"]
        petalLength = data["PetalLengthCm"]
        petalWidth = data["PetalWidthCm"]

        sLength, sWidth, pLength, pWidth = [],[],[],[]
        for index in range(len(Species)):
            if Species[index] == flowerType:
                sLength.append(sepalLength[index])
                sWidth.append(sepalWidth[index])
                pLength.append(petalLength[index])
                pWidth.append(petalWidth[index])
            else:
                continue
        
        if bias == None:
            self.A = self.createA(sLength, sWidth, pLength, pWidth)
        else:
            self.A = self.createA(sLength, sWidth, pLength, pWidth, bias=bias)
        self.b = self.createB(Species.loc[Species==flowerType])
         
    def PLU(self, A, b, bias = False):  
        n = np.shape(A)[0]
        for column in range(len(A[0])):
            L = np.eye(n)
            P = np.eye(n)
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
            b = np.linalg.inv(L).dot(b)
    
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
            coefficients += [x1],[x2],[x3],[x4]
        else:
            x5 = b[4]/A[4][4]
            x4 = (b[3] - A[3][4]*x5)/A[3][3]
            x3 = (b[2] - A[2][4]*x5 - A[2][3]*x4)/A[2][2]
            x2 = (b[1] - A[1][4]*x5 - A[1][3]*x4 - A[1][2]*x3)/A[2][2]
            x1 = (b[0] - A[0][4]*x5 - A[0][3]*x4 - A[0][2]*x3 - A[0][1]*x2)/A[0][0]
            coefficients += [x1],[x2],[x3],[x4],[x5]
        return np.array(coefficients)

    def trainThirdAlgorithm(self, trainDataSet, bias=None):
        train_data = pd.read_csv(trainDataSet)
        sLength = train_data["SepalLengthCm"]
        sWidth = train_data["SepalWidthCm"]
        pLength = train_data["PetalLengthCm"]
        pWidth = train_data["PetalWidthCm"]
        Species = train_data["Species"]

        if bias == None:
            A, b = self.PLU(self.createA(sLength, sWidth, pLength, pWidth), self.createB(Species))
        else:
            A, b = self.PLU(self.createA(sLength, sWidth, pLength, pWidth, bias), self.createB(Species))   
        self.coefficients = self.backSubstitution(A,b)

    def ThirdAlgorithm(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet=0, bias = None):
        if dataSet == 0:
            print(self.coefficients)
            if bias != None:
                flowerClass = np.array([sLength, sWidth, pLength, pWidth,bias]).dot(self.coefficients)
            else:
                flowerClass = np.array([sLength, sWidth, pLength, pWidth]).dot(self.coefficients)
        print(flowerClass)


if __name__ == "__main__":
    a = classifier()
    print("\nA classificação das amostras usando PLU (sem bias) são:\n")
    a.trainThirdAlgorithm("dados_08.csv")
    #a.ThirdAlgorithm(5,2.3,3.3,1)
    #Implementar terceiro algoritmo e fazer questao 2
'''
    #Questão 1 Parte 1
    print("1.1)")
    print("Os coeficientes de aproximação afim com bias são: \n")
    a.trainAlgorithm("dados_08.csv", bias=1)
    print(a.coefficients)
    print("\nSem bias são: \n")
    a.trainAlgorithm("dados_08.csv")
    print(a.coefficients)

    #Questão 1 Parte 2
    print("\n1.2)")
    print("Os coeficientes da classe Iris-setosa (sem bias) são: \n")
    a.Separator("dados_08.csv", "Iris-setosa")
    A,b = a.PLU(a.A, a.b)
    withoutBiasSetosa = a.backSubstitution(a.A, a.b)
    print(withoutBiasSetosa)

    print("\nOs coeficientes da classe Iris-setosa (com bias) são: \n")
    a.Separator("dados_08.csv", "Iris-setosa", bias=1)
    A,b = a.PLU(a.A, a.b)
    withBiasSetosa = a.backSubstitution(a.A, a.b)
    print(withBiasSetosa)

    print("\nOs coeficientes da classe Iris-versicolor (sem bias) são: \n")
    a.Separator("dados_08.csv", "Iris-versicolor")
    A,b = a.PLU(a.A, a.b)
    withoutBiasVersicolor = a.backSubstitution(a.A,a.b)
    print(withoutBiasVersicolor)

    print("\nOs coeficientes da classe Iris-versicolor (com bias) são: \n")
    a.Separator("dados_08.csv", "Iris-versicolor", bias=1)
    A,b = a.PLU(a.A, a.b)
    withBiasVersicolor = a.backSubstitution(A,b)
    print(withBiasVersicolor)
    
    print("\nOs coeficientes da classe Iris-virginica (sem bias) são: \n")
    a.Separator("dados_08.csv", "Iris-virginica")
    A,b = a.PLU(a.A, a.b)
    withoutBiasVirginica = a.backSubstitution(A,b)
    print(withoutBiasVirginica)

    print("\nOs coeficientes da classe Iris-virginica (com bias) são: \n")
    a.Separator("dados_08.csv", "Iris-virginica", bias=1)
    A,b = a.PLU(a.A, a.b)
    withBiasVirginica = a.backSubstitution(A,b)
    print(withBiasVirginica)

    #Questao 2

    #Questão 3


    #Questão 4
    print("\n4)")
    #Algoritmo 1
    print("A classificação das amostras utilizando apenas Minimos Quadrados (sem bias) são:\n")
    a.trainAlgorithm("dados_08.csv")
    a.classifierAlgorithm(5,2.3,3.3,1)
    a.classifierAlgorithm(4.6,3.2,1.4,0.2)
    a.classifierAlgorithm(5,4.4,1.4,0.2)
    a.classifierAlgorithm(6.1,3,4.6,1.4)
    a.classifierAlgorithm(5.9,3,5.1,1.8)
    
    print("\nTeste com todo o dataset (sem bias) utilizando amostra completa (150 flores)")
    a.trainAlgorithm("iris2.csv")
    a.classifierAlgorithm(dataSet="dados_08.csv")

    print("\n")
    print("A classificação das amostras utilizando apenas Minimos Quadrados (com bias) são: \n")
    a.trainAlgorithm("dados_08.csv", bias=1)
    a.classifierAlgorithm(5,2.3,3.3,1,bias=1)
    a.classifierAlgorithm(4.6,3.2,1.4,0.2,bias=1)
    a.classifierAlgorithm(5,4.4,1.4,0.2,bias=1)
    a.classifierAlgorithm(6.1,3,4.6,1.4,bias=1)
    a.classifierAlgorithm(5.9,3,5.1,1.8,bias=1)

    print("\nTeste com todo o dataset (com bias) utilizando amostra completa (150 flores)")
    a.trainAlgorithm("iris2.csv", bias=1)
    a.classifierAlgorithm(dataSet="dados_08.csv",bias=1)

    #Algorito 2
    print("\n")
    print("A classificação das amostras utilizando OneVsAll (sem bias) são: \n")
    a.trainAlgorithm("dados_08.csv", alternativeAlgorithm=True)
    a.OneVsAllAlgorithm(5,2.3,3.3,1)
    a.OneVsAllAlgorithm(4.6,3.2,1.4,0.2)
    a.OneVsAllAlgorithm(5,4.4,1.4,0.2)
    a.OneVsAllAlgorithm(6.1,3,4.6,1.4)
    a.OneVsAllAlgorithm(5.9,3,5.1,1.8)

    print("\nTeste com todo o dataset (sem bias) utilizando amostra completa (150 flores)")
    a.trainAlgorithm("iris2.csv", alternativeAlgorithm=True)
    a.OneVsAllAlgorithm(dataSet="dados_08.csv")

    print("\n")
    print("A classificação das amostras utilizando  OneVsAll (com bias) são: \n")
    a.trainAlgorithm("dados_08.csv", alternativeAlgorithm=True, bias=1)
    a.OneVsAllAlgorithm(5,2.3,3.3,1,bias=1)
    a.OneVsAllAlgorithm(4.6,3.2,1.4,0.2,bias=1)
    a.OneVsAllAlgorithm(5,4.4,1.4,0.2,bias=1)
    a.OneVsAllAlgorithm(6.1,3,4.6,1.4,bias=1)
    a.OneVsAllAlgorithm(5.9,3,5.1,1.8,bias=1)
    
    print("\nTeste com todo o dataset (com bias) utilizando amostra completa (150 flores)")
    a.trainAlgorithm("iris2.csv", alternativeAlgorithm=True, bias=1)
    a.OneVsAllAlgorithm(dataSet="dados_08.csv",bias=1)
'''
    #Algoritmo 3
    