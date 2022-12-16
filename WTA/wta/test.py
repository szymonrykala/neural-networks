import random
from NNutils import Neuron, Measurments, VectorsGenerator, TrainingPoints

# WTA

# Neuron.setMeasurment(Measurments.geometricMeasure)

# inputVectorLength = 3
# neuronsCount = 5

# neurons = [Neuron(5) for _ in range(inputVectorLength)]

# for n in neurons: print(n)

# generator = VectorsGenerator(3,2,2)

# vectors = generator.run()
# vectors = TrainingPoints.loadFromFile('temp.json')

# print(vectors.points)
# vectors.saveToFile('./temp1.json')

# vectors.shuffle()

# print(vectors.points)


def runGenerator():
    generator = VectorsGenerator(3,0,2)

    vectors = generator.run(0,100,1)
    print(vectors.points)
    vectors.saveToFile('temp.json')

runGenerator()