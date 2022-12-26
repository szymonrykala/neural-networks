import json
import logging
from NNutils import Neuron, Measurments, Saveable, TrainingPoints

logging.basicConfig(level=logging.INFO,
                    format='%(name)s:%(levelname)s %(message)s')

WTA_FILE = './WTA.json'


class WTA():
    P: float = 0.5
    LEARN: bool = True
    MEASURE: Measurments = Measurments.metropolitanMeasure

    def __init__(self, neurons: list[Neuron]) -> None:
        self.__neurons: Saveable = Saveable(neurons)
        self.logger = logging.getLogger(__class__.__name__)

        self.__lastWinnerIndex = -1

    @property
    def neurons(self):
        return self.__neurons.store

    @classmethod
    def build(cls, inputVectorLength: int, neuronsCount: int):
        Neuron.setMeasurment(cls.MEASURE)
        Neuron.P = cls.P

        return cls([
            Neuron(inputVectorsCount=inputVectorLength)
            for _ in range(neuronsCount)
        ])

    @classmethod
    def buildFromFile(cls, filePath: str):
        Neuron.setMeasurment(cls.MEASURE)
        Neuron.P = cls.P
        savedNeuronsWages = Saveable.loadFromFile(filePath).store
        return cls([
            Neuron(initWages=wages)
            for wages in savedNeuronsWages
        ])

    def save(self, filePath: str):
        self.__neurons.saveToFile(filePath)

    def logNeurons(self):
        for i, n in enumerate(self.neurons):
            self.logger.info(f'{i}. {n}')

    def process(self, vector: list | tuple) -> tuple[int, float]:
        '''
        param:
            vector:list|tuple - vector to be processed by the network,
        returns:
            tuple(triggered_neuron_index, similarity_value)

        updates neurons weights only if LEARN property is set to True
        '''
        similarity = [
            (index, neuron.getSimilarityTo(vector))
            for index, neuron in enumerate(self.__neurons.store)
        ]
        similarity.sort(key=lambda s: s[1], reverse=True)
        self.logger.debug(similarity)

        # choose the winner
        choosedIndex = 0  # biggest similarity
        (index, sim) = similarity[choosedIndex]
        # while self.LEARN and self.__lastWinnerIndex == index:
        #     self.logger.debug(f'skipping {index}; sim: {sim}')
        #     (index, sim) = similarity[choosedIndex+1]

        # if self.LEARN and self.__lastWinnerIndex != index:
        self.__neurons.store[index].updateWages(vector)

        self.logger.debug(f'winner: {index}; sim: {sim}')
        self.__lastWinnerIndex = index

        return (index, sim)


WTA.P = 0.4
WTA.LEARN = True
WTA.MEASURE = Measurments.metropolitanMeasure


def train():
    DESIRED_SIMILARITY = 4.5
    MAX_INTERATIONS = 10_000

    wta = WTA.build(2, 3)
    wta.logNeurons()

    vectors = TrainingPoints.loadFromFile('./vectors.json')

    runResults = [(-1, 999)]
    currentIter = 0
    while (
        any([res[1] > DESIRED_SIMILARITY for res in runResults])
        and currentIter < MAX_INTERATIONS
    ):

        runResults = [
            wta.process(vector)
            for vector in vectors.points
        ]

        vectors.shuffle()
        currentIter += 1

    print(currentIter)
    print(runResults)
    wta.logNeurons()
    wta.save(WTA_FILE)


class NetworkTestResults(Saveable):
    '''
    {
        'neuronIndex': [
                {
                    'sim': float,
                    'vector': arr[floats]
                },
                ...
            ],
        ...
    }
    '''

    def __init__(self) -> None:
        super()
        self.data = {}

    def add(self, neuronIndex, sim, vector) -> None:
        if neuronIndex in self.data:
            self.data[neuronIndex].append({
                'sim': sim,
                'vector': vector
            })
        else:
            self.data[neuronIndex] = [
                {
                    'sim': sim,
                    'vector': vector
                }
            ]
            
    def dump(self):
        return json.dumps(self.data, indent=4)


def test():
    store = NetworkTestResults()
    
    WTA.LEARN = False
    wta = WTA.buildFromFile(WTA_FILE)
    wta.logNeurons()

    vectors = TrainingPoints.loadFromFile('./vectors.json')

    for vector in vectors.points:
        neuron, sim = wta.process(vector)
        store.add(neuron, sim, vector)


    print(store.dump())


# test()
train()
