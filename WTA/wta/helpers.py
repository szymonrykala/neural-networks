from NNutils import Neuron, Measurments, Saveable, VectorsGenerator, TrainingPoints

NEURONS_FILE = 'WTANeurons.json'
VECTORS_FILE = 'vectors.json'


def createNeurons(
        count:int,
        inputVectorLength:int, 
        measure:Measurments=Measurments.geometricMeasure,
        p:float = 0.2
    ):
    Neuron.setMeasurment(measure)
    Neuron.P = p
    return [
        Neuron(inputVectorsCount=inputVectorLength) 
        for _ in range(count)
    ]


def loadNeurons(inputVectorLength:int, file:str=NEURONS_FILE):
    savedNeuronsWages = Saveable.loadFromFile(file)
    return [
        Neuron(inputVectorsCount=inputVectorLength, initWages=wages)
        for wages in savedNeuronsWages.store 
    ]


def generateVectors(
    points:int=3, radialPoints:int=2, dimension:int=2,
    radius:int=1, min:int=0, max:int=100,
    file:str=VECTORS_FILE
):
    gen = VectorsGenerator(points,radialPoints, dimension)
    vectors = gen.run(min,max,radius)
    vectors.saveToFile(file)


def loadVectors(file:str=VECTORS_FILE):
    return TrainingPoints.loadFromFile(file)