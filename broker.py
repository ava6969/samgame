from typing import List
from common import Account, Position
from dataloader import Dataloader


class Broker:
    """
     https://alpaca.markets/docs/api-documentation/api-v2/account/
    """

    def __init__(self, account: Account):
        # attributes
        self.account = account
        self.positions = List[Position]
        self.dataloader = Dataloader()

    def start(self, tech_indicators):
        """
        starts new session
        :param tech_indicators:
        :return:
        """
        pass

    def peek(self, step_amount):
        """
        dont update time just check next
        :param step_amount:
        :return:
        """
        pass

    def step(self, step_amount):
        """
        step to next timestep[1 min, 1 week, ]
        :param step_amount how many steps to go into data
        :return:
        """

    def __repr__(self):
        account = ''
        account += f'start_capital: {self.account.cash}\n'
        return account


if __name__ == '__main__':
    marcus_pfolio = Broker(Account(account_number=32425, buying_power=1000, cash=1000, daytrade_count=0,
                                   daytrading_buying_power=0, equity=1000, id=0, initial_margin=1000, last_equity=1000,
                                   ))
    print(marcus_pfolio)

