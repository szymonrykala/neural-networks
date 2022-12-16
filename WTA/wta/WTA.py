import logging
from NNutils import Neuron, Measurments, VectorsGenerator, TrainingPoints

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')


def nprint(lvlFunction=logging.info):
    for i, n in enumerate(neurons):
        lvlFunction(f'{i}. {n}')
    lvlFunction('-')

epochsCount = 1_000
# WTA


# learning speed p
P = 0.99

Neuron.setMeasurment(Measurments.metropolitanMeasure)
Neuron.P = P

inputVectorLength = 2
neuronsCount = 9

neurons = [Neuron(inputVectorLength) for _ in range(neuronsCount)]
nprint()

vectors = TrainingPoints.loadFromFile('temp.json')


def main():
    lastWinner = 0
    for _ in range(epochsCount):
        print(f'epoch: {_}')
        for vector in vectors.points:

            similarity = [
                (index, neuron.getSimilarityTo(vector))
                for index, neuron in enumerate(neurons)
            ]
            similarity.sort(key=lambda s: s[1], reverse=True)

            #choose the winner
            choosedIndex = 0 # biggest similarity
            (index, sim) = similarity[choosedIndex]
            # while lastWinner == index:
            #     logging.debug(f'skipping {index}; sim: {sim}')
            #     (index, sim) = similarity[choosedIndex+1]

            if lastWinner != index:
                neurons[index].updateWages(vector)

            logging.debug(f'winner: {index}; sim: {sim}')
            lastWinner = index

            #neuron wages update
            # nprint(logging.debug)
            
        vectors.shuffle() 

main()

nprint()