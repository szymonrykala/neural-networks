import json
from math import sqrt
import random


class Measurments():
    @classmethod
    def metropolitanMeasure(cls, w: float, v: float) -> float:
        return sqrt(abs(w - v))

    @classmethod
    def geometricMeasure(cls, w: float, v: float) -> float:
        return sqrt((w-v)*(w-v))


class Neuron():
    P = 0.1

    def __init__(self, inputVectorsCount: int, initWages: list = None) -> None:
        if initWages:
            self.wages = initWages
        else:
            self.wages = [
                self.__getRandWage()
                for _ in range(inputVectorsCount)
            ]

    def __str__(self) -> str:
        return str(self.wages)

    @classmethod
    def setMeasurment(cls, measurement: Measurments) -> None:
        cls.__countMeasure = measurement

    @classmethod
    def __countMeasure(cls, w: float, v: float) -> float:
        raise NotImplementedError

    def getSimilarityTo(self, vector: list):
        return sum(self.__countMeasure(w, v) for w, v in zip(self.wages, vector))

    def __getRandWage(self):
        return random.randrange(1, 100)

    def updateWages(self, inputVector, A=1):
        for inputNumber in range(len(self.wages)):
            self.wages[inputNumber] = self.wages[inputNumber] + self.P * \
                (inputVector[inputNumber] - self.wages[inputNumber]) * A


class Saveable():
    FILE_NAME = './data.json'

    def __init__(self, storeData) -> None:
        self.store = storeData

    def updateStore(self, data):
        self.store = data

    @classmethod
    def loadFromFile(cls, fileName: str = None):
        if not fileName:
            fileName = cls.FILE_NAME

        with open(fileName) as file:
            data = file.read()
            points = json.loads(data)
            file.close()
            return cls(points)

    def saveToFile(self, fileName: str = None):
        if not fileName:
            fileName = self.FILE_NAME

        with open(fileName, 'w+') as file:
            data = json.dumps(self.points)
            file.write(data)
            file.close()


class TrainingPoints(Saveable):

    def __init__(self, points: list) -> None:
        super().__init__(points)
        self.points: list = points

    def shuffle(self):
        random.shuffle(self.points)


class VectorsGenerator():
    __max: float = 10
    __min: float = 0
    __radius = 2

    def __init__(
        self,
        mainPontsCount: int,
        radialPointsCount: int,
        dimensions: int = 2
    ) -> None:
        self.mainPointsCount: int = mainPontsCount
        self.radialPointsCount: int = radialPointsCount
        self.dimensions: int = dimensions
        self.points: list = []

    def __getRandomPoint(self, min: float, max: float) -> float:
        return round((random.random() * (max-min) ) + min,2)

    def __getRadialPoint(self, point: list[float]) -> list[float]:
        return [
            self.__getRandomPoint(
                point[dim]-self.__radius,
                point[dim]+self.__radius
            )
            for dim in range(self.dimensions)
        ]

    def __generateMainPoints(self) -> None:
        self.points = [
            [
                self.__getRandomPoint(self.__min, self.__max)
                for _ in range(self.dimensions)
            ]
            for _ in range(self.mainPointsCount)
        ]

    def __generateRadialPoints(self) -> None:
        radial = []

        for point in self.points:
            radial.extend([
                self.__getRadialPoint(point)
                for _ in range(self.radialPointsCount)
            ])

        self.points.extend(radial)

    def run(self, min: int = __min, max: int = __max, radius: int = __radius) -> TrainingPoints:
        self.__min = min
        self.__max = max
        self.__radius = radius

        self.__generateMainPoints()
        self.__generateRadialPoints()

        return TrainingPoints(self.points)
