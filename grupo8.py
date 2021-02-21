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
                        aux = 0
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

    def OneVsAllAlgorithm(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet = 0, bias = None): #Algoritmo de Classificação usando Um Contra Todos
        if dataSet == 0:
            if bias!= None:
                isSetosa = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(self.setosaClassifiers)[0]
                isVersicolor = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(self.versicolorClassifiers)[0]
                isVirginica = np.array([sLength, sWidth, pLength, pWidth, bias]).dot(self.virginicaClassifiers)[0]
            else:
                isSetosa = np.array([sLength, sWidth, pLength, pWidth]).dot(self.setosaClassifiers)[0]
                isVersicolor = np.array([sLength, sWidth, pLength, pWidth]).dot(self.versicolorClassifiers)[0]
                isVirginica = np.array([sLength, sWidth, pLength, pWidth]).dot(self.virginicaClassifiers)[0]
        
            print("\n\nAs porcentagens são: \n")
            print("Setosa: "+str(round(isSetosa*100,1))+"%")
            print("Versicolor: "+str(round(isVersicolor*100,1))+"%")
            print("Virginica: "+str(round(isVirginica*100,1))+"%")

            highestCoef = self.highest(isSetosa, isVersicolor, isVirginica)
            if highestCoef == isSetosa:
                print("É uma Iris Setosa")
            elif highestCoef == isVirginica:
                print("É uma Iris Virginica")
            elif highestCoef == isVersicolor:
                print("É uma Iris Versicolor")
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
            print("Acurácia de "+str(aux1/aux2))
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
                if round(flowerClass[item][0]) <= 0:
                    response = "Iris-setosa"
                elif round(flowerClass[item][0]) == 1:
                    response = "Iris-versicolor"
                else:
                    response = "Iris-virginica"
                aux1, aux2 = self.Acurrancier(item, response, Species, aux1, aux2)
            print("Acurácia de "+str(aux1/aux2))
        return aux1/aux2
         
    def SeparatorSetosa(self, dataSet, bias=None):
        data = pd.read_csv(dataSet)
        Species = data["Species"]
        sepalLength = data["SepalLengthCm"].loc[Species=="Iris-setosa"]
        sepalWidth = data["SepalWidthCm"].loc[Species=="Iris-setosa"]
        petalLength = data["PetalLengthCm"].loc[Species=="Iris-setosa"]
        petalWidth = data["PetalWidthCm"].loc[Species=="Iris-setosa"]

        if bias == None:
            A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth)
        else:
            A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth, bias=bias)
        b = self.createB(Species.loc[Species=="Iris-setosa"])
        return A, b
         
    def SeparatorVersicolor(self, dataSet, bias=None):
        data = pd.read_csv(dataSet)
        Species = data["Species"]
        sepalLength = data["SepalLengthCm"].loc[Species=="Iris-versicolor"]
        sepalWidth = data["SepalWidthCm"].loc[Species=="Iris-versicolor"]
        petalLength = data["PetalLengthCm"].loc[Species=="Iris-versicolor"]
        petalWidth = data["PetalWidthCm"].loc[Species=="Iris-versicolor"]

        if bias == None:
            A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth)
        else:
            A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth, bias=bias)
        b = self.createB(Species.loc[Species=="Iris-versicolor"])
        return A, b

    def SeparatorVirginica(self, dataSet, bias=None):
        data = pd.read_csv(dataSet)
        Species = data["Species"]
        sepalLength = data["SepalLengthCm"].loc[Species=="Iris-virginica"]
        sepalWidth = data["SepalWidthCm"].loc[Species=="Iris-virginica"]
        petalLength = data["PetalLengthCm"].loc[Species=="Iris-virginica"]
        petalWidth = data["PetalWidthCm"].loc[Species=="Iris-virginica"]

        if bias == None:
            A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth)
        else:
            A = self.createA(sepalLength, sepalWidth, petalLength, petalWidth, bias=bias)
        b = self.createB(Species.loc[Species=="Iris-virginica"])

        return A, b

if __name__ == "__main__":
    a = classifier() #iniciando a classe
    #print(a.SeparatorSetosa("dados_08.csv"))
    print(a.SeparatorVersicolor("dados_08.csv"))
    print(a.SeparatorVirginica("dados_08.csv"))
    '''
    #Questão 1 Parte 1
    print("Os coeficientes de aproximação afim são: \n")
    print(a.leastSquares())

    #Questão 1 Parte 2


    #Questão 4
    print("\n")
    print("A classificação das amostras são:")
    a.Classifier(5,2.3,3.3,1)
    a.Classifier(4.6,3.2,1.4,0.2)
    a.Classifier(5,4.4,1.4,0.2)
    a.Classifier(6.1,3,4.6,1.4)
    a.Classifier(5.9,3,5.1,1.8)
    a.Classifier("iris.csv")

'''
    
    #fazer questao 1 parte 2
  