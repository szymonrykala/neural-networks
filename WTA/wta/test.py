import random
from NNutils import Neuron, Measurments, Saveable, VectorsGenerator, TrainingPoints
from helpers import VECTORS_FILE

# WTA

Neuron.setMeasurment(Measurments.geometricMeasure)

inputVectorLength = 3
neuronsCount = 2

# neurons = [Neuron(inputVectorLength) for _ in range(neuronsCount)]


# print(neurons)

# loadedNeurons = Saveable.loadFromFile('WTANeurons.json').store


# saveableNeurons = Saveable(neurons)

# saveableNeurons.saveToFile('tempNeurons.json')

# for n in neurons: print(n)

# generator = VectorsGenerator(3,2,2)

# vectors = generator.run()
# vectors = TrainingPoints.loadFromFile('temp.json')

# print(vectors.points)
# vectors.saveToFile('./temp1.json')

# vectors.shuffle()

# print(vectors.points)


def runGenerator():
    generator = VectorsGenerator(3,10,2)

    vectors = generator.run(min=0,max=100,radius=0.5)
    print(vectors.points)
    vectors.saveToFile(VECTORS_FILE)

runGenerator()
