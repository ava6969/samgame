from common import *


class Portfolio:
    """
    constraint transaction cost = 0
    slippage cost = 0
    """
    def __init__(self):
        self.positions = List[Position]

    def place_order(self, order: Order, stock_price_dollars:float, account: Account):
        if order.side == 'buy':
            self.buy_stocks(order, stock_price_dollars)
        elif order.side == 'sell':
            self.sell_stocks(order)
        else:
            raise ValueError('position side can only be buy and sell')
        self.update_account(account)

    def buy_stocks(self, order, stock_price_dollars, account: Account):
        """
        update account
        :param order:
        :param market_value:
        :return:
        """
        # order.qty, stock_price_dollars
        # account.cash

        # cash - qty * stock_price_dollars
        # update cash left
        # update porfolio value
        pass

    def sell_stocks(self, order, account):
        """
        update account
        :param order:
        :return:
        """
        # order.qty, stock_price_dollars
        # account.cash

        # cash - qty * stock_price_dollars
        # update cash left
        # update porfolio value
        pass

    def update_account(self, account):
        pass
