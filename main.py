import backtester
import broker


class MathCalculator:
    def __init__(self):
        self.x = 2
        self.y = 4

    def add(self):
        return self.x+self.y

    def sub(self):
        return self.x - self.y

    def mul(self):
        return self.x * self.y


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


if __name__ == '__main__':
    print(add(2, 3))
    print(sub(2, 3))
    print(mul(2, 3))

    print('from class')
    # classes
    calc = MathCalculator()
    print(calc.add())
    print(calc.mul())
    print(calc.sub())