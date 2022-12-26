import logging
from NNutils import Neuron, Measurments, Saveable, VectorsGenerator, TrainingPoints
from helpers import createNeurons, loadVectors

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


def nprint(lvlFunction=logging.info):
    for n in enumerate(neurons):
        lvlFunction(f'{n[0]}. {n[1]}')
    lvlFunction('-')

epochsCount = 10
# WTA


inputVectorLength = 2

vectors = loadVectors()

neurons = createNeurons(
    count=6,
    inputVectorLength=len(vectors.points[0]),
    measure=Measurments.metropolitanMeasure,
    p=0.6
)

nprint()


def main(neurons:list[Neuron])->Saveable:
    lastWinner = 0
    for _ in range(epochsCount):
        logging.debug(f'epoch: {_}')
        for vector in vectors.points:

            similarity = [
                (index, neuron.getSimilarityTo(vector))
                for index, neuron in enumerate(neurons)
            ]
            similarity.sort(key=lambda s: s[1], reverse=True)

            #choose the winner
            choosedIndex = 0 # biggest similarity
            (index, sim) = similarity[choosedIndex]
            while lastWinner == index:
                logging.debug(f'skipping {index}; sim: {sim}')
                (index, sim) = similarity[choosedIndex+1]

            if lastWinner != index:
                neurons[index].updateWages(vector)

            logging.debug(f'winner: {index}; sim: {sim}')
            lastWinner = index

            #neuron wages update
            # nprint(logging.debug)
            
        vectors.shuffle()
    
    return Saveable(neurons)


saveabletrainedNeurons = main(neurons)

saveabletrainedNeurons.saveToFile('WTANeurons.json')

nprint()