from typing import Dict, List
from datamodel import Order, ProsperityEncoder, TradingState, Symbol, OrderDepth
import math
import json
import numpy as np
import pandas as pd

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]]) -> None:
        logs = self.logs
        if logs.endswith("\n"):
            logs = logs[:-1]

        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.state = None
        self.orders = {}
        self.logs = ""

logger = Logger()

class Trader:

    def __init__(self):
        self.past_data = {'BANANAS': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'COCONUTS': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'PINA_COLADAS': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'PEARLS': {'position': []},
                          'BERRIES': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'DIVING_GEAR': {'bid_ask_avg': [], 'mid_price': [], 'position': []}}
        self.bananas_limit = self.pearls_limit = 20
        self.berries_limit = 250
        self.gear_limit = 50
        self.coconuts_limit = 600
        self.pina_colada_limit = 300
        self.T = 1000000
        self.T = 100000

    def get_weight_price(self, depth):
        avg_ask = sum([depth.sell_orders[key] * key for key in depth.sell_orders.keys()])/sum(depth.sell_orders.values())
        avg_bid = sum([depth.buy_orders[key] * key for key in depth.buy_orders.keys()])/sum(depth.buy_orders.values())
        return (avg_ask + avg_bid) / 2
    
    def find_bids(self, weighted, depth, gap, n):
        asks = list(depth.sell_orders.keys())
        asks.sort()
        bid = ask = 0
        for price in asks:
            if price >= weighted:
                ask = price - gap
                break
        bids = list(depth.buy_orders.keys())
        bids.sort(reverse = True)
        for price in bids:
            if price <= weighted:
                bid = price + gap
                break
        if bid == 0:
            bid = weighted + n
        if ask == 0:
            ask = weighted + n
        return bid, ask
    
    def sma(self, duration, product):
        l = self.past_data[product]['mid_price']
        return list(pd.Series(l).ewm(duration).mean())[-2:]

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        # Initialize the method output dict as an empty dict
        result = {}

        c_weighted_price = self.get_weight_price(state.order_depths['COCONUTS'])
        p_weighted_price = self.get_weight_price(state.order_depths['PINA_COLADAS'])
        signal = 1.875 * c_weighted_price - p_weighted_price      
        for product in state.order_depths.keys():
            curr_pos = 0
            t = state.timestamp
            if product in state.position.keys():
                curr_pos = state.position[product]
            self.past_data[product]['position'].append(curr_pos)
            if product == 'BERRIES':
                did_buy = did_sell = 0
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                acceptable_price = self.get_weight_price(order_depth)
                self.past_data[product]['mid_price'].append(acceptable_price)
                if product in state.position.keys():
                    pos = state.position[product]
                else: pos = 0
                b, a = self.find_bids(acceptable_price, order_depth, 1, 3)
                tilt = (curr_pos/250) * 2
                long = self.sma(int(self.T * 0.0005), product)
                short = self.sma(int(self.T * 0.0002), product)
                if t >= self.T * 0.4 and t < self.T * 0.6:
                    if short[1] >= long[1]:
                        orders.append(Order(product, acceptable_price + 1, (self.berries_limit-pos)/2))
                        orders.append(Order(product, acceptable_price + 2, 3*(self.berries_limit-pos)/8))
                        orders.append(Order(product, acceptable_price + 3, (self.berries_limit-pos)/8))
                        did_buy = self.berries_limit-pos
                if t >= self.T * 0.6 and t < self.T * 0.7:
                    if short[1] <= long[1] or short[1] > short[0]:
                        orders.append(Order(product, acceptable_price - 1, (-self.berries_limit-pos)/2))
                        orders.append(Order(product, acceptable_price - 2, 3*(-self.berries_limit-pos)/8))
                        orders.append(Order(product, acceptable_price - 3, (-self.berries_limit-pos)/8))
                        did_sell =-self.berries_limit-pos
                if t >= self.T * 0.7 and t <= self.T * 0.8:
                    orders.append(Order(product, acceptable_price + 1, (-pos)/2))
                    orders.append(Order(product, acceptable_price + 2, 3*(-pos)/8))
                    orders.append(Order(product, acceptable_price + 3, (-pos)/8))
                    did_buy = -pos

                if t < self.T * 0.3 or t > self.T * 0.8:
                    if did_buy + pos < self.berries_limit:
                        orders.append(Order(product, b + max(0, tilt), self.berries_limit-pos-did_buy))
                    if did_sell + pos > -self.berries_limit:
                        orders.append(Order(product, a - min(0, tilt), -self.berries_limit-pos-did_sell))
                result[product] = orders
            if product == 'PEARLS':
                did_buy = did_sell = 0
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                acceptable_price = 10000
                if product in state.position.keys():
                    pos = state.position[product]
                else: pos = 0

                if len(order_depth.sell_orders) > 0:
                    asks = list(order_depth.sell_orders.keys())
                    asks.sort()
                    for i in asks:
                        if i <= acceptable_price:
                            ask_vol = min(-order_depth.sell_orders[i], self.pearls_limit - pos - did_buy)
                            if ask_vol > 0:
                                orders.append(Order(product, i, ask_vol)) # postitive volume
                                did_buy += ask_vol

                if len(order_depth.buy_orders) > 0:
                    bids = list(order_depth.buy_orders.keys())
                    bids.sort(reverse = True)
                    for i in bids:
                        if i >= acceptable_price:
                            bid_vol = max(-order_depth.buy_orders[i], -self.pearls_limit - pos - did_sell)
                            if bid_vol < 0:
                                orders.append(Order(product, i, bid_vol)) # postitive volume
                                did_sell += bid_vol
                n = 3 if order_depth.buy_orders[bids[0]] > 12 else 4
                m = 3 if order_depth.sell_orders[asks[0]] > 12 else 4
                if did_buy + pos < self.pearls_limit:
                    orders.append(Order(product, acceptable_price - n, self.pearls_limit-pos-did_buy))
                if did_sell + pos > -self.pearls_limit:
                    orders.append(Order(product, acceptable_price + m, -self.pearls_limit-pos-did_sell))
                result[product] = orders

            if product == 'BANANAS':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                weighted_price = self.get_weight_price(order_depth)
                r = weighted_price - (curr_pos/20) * 2
                delta = 4
                bid_price = r-delta/2
                ask_price = r+delta/2
                orders.append(Order(product, bid_price, self.bananas_limit - curr_pos))
                orders.append(Order(product, ask_price, -curr_pos - self.bananas_limit))
                result[product] = orders


        logger.flush(state, result)
        return result