import numpy as np 
import pandas as pd

data = pd.read_csv("dados_08.csv")

class classifier():
    def __init__(self):
        self.SepalLength = data["SepalLengthCm"]
        self.SepalWidth = data["SepalWidthCm"]
        self.PetalLength = data["PetalLengthCm"]
        self.PetalWidth = data["PetalWidthCm"]
        self.species = data["Species"]
    
    def createB(self, datas):
        array = []
        for linha in datas:
            if type(linha) == str:
                if linha == "Iris-setosa":
                    aux = 0
                elif linha == "Iris-versicolor":
                    aux = 1
                else:
                    aux = 2
                array.append([aux])
        return array

    def createA(self, sLength, sWidth, pLength, pWidth):
        matrix = []
        for index in range(len(sLength)):
            matrix.append([sLength[index], sWidth[index], pLength[index], pWidth[index]])
        return matrix
    
    def leastSquares(self):
        A = np.array(self.createA(self.SepalLength, self.SepalWidth, self.PetalLength, self.PetalWidth))
        b = np.array(self.createB(self.species))
        x = np.linalg.inv((A.transpose().dot(A))).dot(A.transpose()).dot(b)
        return x

    def Acurrancier(self, index, response, hits = 0, tests = 0):
        if self.species[index] == response:
            hits+=1
            tests+=1
        else:
            tests+=1
        return hits,tests

    def Classifier(self, sLength=0, sWidth=0, pLength=0, pWidth=0, dataSet = 0):
        if dataSet == 0:
            flowerClass = np.array([sLength, sWidth, pLength, pWidth]).dot(self.leastSquares())
            if round(flowerClass[0]) <= 0:
                print("Iris-setosa")
            elif round(flowerClass[0]) == 1:
                print("Iris-versicolor")
            else:
                print("Iris-virginica")
        else:
            flowerClass = np.array(dataSet).dot(self.leastSquares())
            aux1 = 0
            aux2 = 0
            for item in range(len(flowerClass)):
                if round(flowerClass[item][0]) <= 0:
                    response = "Iris-setosa"
                elif round(flowerClass[item][0]) == 1:
                    response = "Iris-versicolor"
                else:
                    response = "Iris-virginica"
                #print(response)
                aux1, aux2 = self.Acurrancier(item, response, aux1, aux2)
            print("Acurácia de "+str(aux1/aux2))

    def printer(self):
        #self.Classifier(5.2,4,4,0.7)
        self.Classifier(dataSet=self.createA(self.SepalLength, self.SepalWidth, self.PetalLength, self.PetalWidth))

if __name__ == "__main__":
    a = classifier()
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


    

    #separar em dataset para teste e classificação
    #fazer questao 1 parte 2
    # implementar algoritmo 1 contra todos