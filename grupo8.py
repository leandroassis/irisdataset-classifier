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
                    if linha == "Setosa":
                        aux = 0
                    elif linha == "Versicolor":
                        aux = 1
                    else:
                        aux = 2
                    array.append([aux])
        return np.array(array)

    def createA(self, sLength, sWidth, pLength, pWidth, bias = False):
        matrix = []
        for index in range(len(sLength)):
            if bias:
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

    def isSetosa(self, sLength, sWidth, pLength, pWidth, Species, bias=0):
        A = self.createA(sLength, sWidth, pLength, pWidth)
        b = self.createB(Species, "Iris-setosa")
        classifierIsSetosa = self.leastSquares(A,b)
        return classifierIsSetosa

    def isVersicolor(self, sLength, sWidth, pLength, pWidth, Species, bias=0):
        A = self.createA(sLength, sWidth, pLength, pWidth)
        b = self.createB(Species, "Iris-versicolor")
        classifierIsVersicolor = self.leastSquares(A,b)
        return classifierIsVersicolor
    
    def isVirginica(self, sLength, sWidth, pLength, pWidth, Species, bias=0):
        A = self.createA(sLength, sWidth, pLength, pWidth)
        b = self.createB(Species, "Iris-virginica")
        classifierIsVirginica = self.leastSquares(A,b)
        return classifierIsVirginica

    def trainAlgorithm(self, trainDataSet, alternativeAlgorithm = False, bias = 0):
        train_data = pd.read_csv(trainDataSet)
        sLength = train_data["SepalLengthCm"]
        sWidth = train_data["SepalWidthCm"]
        pLength = train_data["PetalLengthCm"]
        pWidth = train_data["PetalWidthCm"]
        Species = train_data["Species"]

        if alternativeAlgorithm:
            if bias:
                self.setosaClassifiers = self.isSetosa(sLength, sWidth, pLength, pWidth, Species, bias)
                self.versicolorClassifiers = self.isVersicolor(sLength, sWidth, pLength, pWidth, Species, bias)
                self.virginicaClassifiers =self.isVirginica(sLength, sWidth, pLength, pWidth, Species, bias)
            else:
                self.setosaClassifiers = self.isSetosa(sLength, sWidth, pLength, pWidth, Species)
                self.versicolorClassifiers = self.isVersicolor(sLength, sWidth, pLength, pWidth, Species)
                self.virginicaClassifiers =self.isVirginica(sLength, sWidth, pLength, pWidth, Species)
        else:
            A = self.createA(sLength, sWidth, pLength, pWidth)
            b = self.createB(Species)
            self.coefficients = self.leastSquares(A,b)

    def OneVsAllAlgorithm(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet = 0): #Algoritmo de Classificação usando Um Contra Todos
        if dataSet == 0:
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
                isSetosa = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item]]).dot(self.setosaClassifiers)[0]
                isVersicolor = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item]]).dot(self.versicolorClassifiers)[0]
                isVirginica = np.array([sLength[item], sWidth[item], pLength[item], pWidth[item]]).dot(self.virginicaClassifiers)[0]
                
                highestCoef = self.highest(isSetosa, isVirginica, isVersicolor)

                if isSetosa == highestCoef:
                    response = "Iris-setosa"
                elif isVirginica == highestCoef:
                    response = "Iris-virginica"
                elif isVersicolor == highestCoef:
                    response = "Iris-versicolor"
                print(response, Species[item])
                if response == Species[item]:
                    a+=1
                    print("acertou")
                else:
                    print("errou")
                aux1, aux2 = self.Acurrancier(item, response, Species, aux1, aux2)
            print(a)
            print("Acurácia de "+str(aux1/aux2))

    def classifierAlgorithm(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet = 0):
        if dataSet == 0:
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

            flowerClass = self.createA(sepalLength, sepalWidth, petalLength, petalWidth).dot(self.coefficients)
            aux1 = 0
            aux2 = 0
            for item in range(len(flowerClass)):
                if round(flowerClass[item][0]) <= 0:
                    response = "Setosa"
                elif round(flowerClass[item][0]) == 1:
                    response = "Versicolor"
                else:
                    response = "Virginica"
                #print(response)
                aux1, aux2 = self.Acurrancier(item, response, aux1, aux2)
            print("Acurácia de "+str(aux1/aux2))

if __name__ == "__main__":
    a = classifier() #iniciando a classe
    a.trainAlgorithm("dados_08.csv", alternativeAlgorithm=True) #treinando o algoritmo com um dataset selecionado
    a.OneVsAllAlgorithm(dataSet="dados_08.csv")
    a.classifierAlgorithm()
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
  