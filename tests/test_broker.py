from collections import defaultdict
import pytest
from broker import Broker, Account

b = Broker(Account(12345, 1000, 0, 0, 0, defaultdict(lambda: 0), False), 500)


