class PortfolioManager:
    def __init__(self, start_capital):
        # attribute
        self.start_capital = start_capital
        self.balance = start_capital
        self.stock_owned = 0

    def increase_balance(self, amount):
        self.balance = self.balance + amount

    def buy_stocks(self, qty):
        self.stock_owned += qty

    def sell_stocks(self, qty):
        self.stock_owned -= qty

    def print_account(self):
        print('start_capital:', self.start_capital)
        print('balance:', self.balance)


if __name__ == '__main__':
    marcus_pfolio = PortfolioManager(1000)
    marcus_pfolio.print_account()

    marcus_pfolio.increase_balance(100)
    marcus_pfolio.print_account()

    damola_pfolio =  PortfolioManager(10000000)
    damola_pfolio.print_account()




