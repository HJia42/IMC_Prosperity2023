import json
from datamodel import Order, ProsperityEncoder, Symbol, TradingState, Trade, OrderDepth
from typing import Any, Dict, List
import math
import numpy as np
import pandas as pd

class Logger:
    # Set this to true, if u want to create
    # local logs
    local: bool 
    # this is used as a buffer for logs
    # instead of stdout
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        output = json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True)
        if self.local:
            self.local_logs[state.timestamp] = output
        print(output)

        self.logs = ""

    def compress_state(self, state: TradingState) -> dict[str, Any]:
        listings = []
        for listing in state.listings.values():
            listings.append([listing["symbol"], listing["product"], listing["denomination"]])

        order_depths = {}
        for symbol, order_depth in state.order_depths.items():
            order_depths[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return {
            "t": state.timestamp,
            "l": listings,
            "od": order_depths,
            "ot": self.compress_trades(state.own_trades),
            "mt": self.compress_trades(state.market_trades),
            "p": state.position,
            "o": state.observations,
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.buyer,
                    trade.seller,
                    trade.price,
                    trade.quantity,
                    trade.timestamp,
                ])

        return compressed

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

# This is provisionary, if no other algorithm works.
# Better to loose nothing, then dreaming of a gain.
class Trader:

    logger = Logger(local=True)
    def __init__(self):
        self.past_data = {'BANANAS': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'COCONUTS': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'PINA_COLADAS': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'PEARLS': {'position': []},
                          'DOLPHIN_SIGHTINGS':{'position':[]},
                          'BERRIES': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'DIVING_GEAR': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'UKULELE': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'DIP': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'PICNIC_BASKET': {'bid_ask_avg': [], 'mid_price': [], 'position': []},
                          'BAGUETTE': {'bid_ask_avg': [], 'mid_price': [], 'position': []}}
        self.bananas_limit = self.pearls_limit = 20
        self.berries_limit = 250
        self.gear_limit = 50
        self.coconuts_limit = 600
        self.pina_colada_limit = self.dip_limit = 300
        self.baguette_limit = 150
        self.ukulele_limit = self.picnic_limit = 70
        self.T = 1000000
        self.seen = []
        self.buying = self.selling = False
        self.timeevent = 0
        self.short_price = self.long_price = 0
        self.olivia_b = 0
        self.did_buy_b = self.did_sell_b = False
        self.did_buy_at_b = self.did_sell_at_b = 0
        self.olivia_pair = 0
        self.did_buy_p = self.did_sell_p = self.did_buy_d = self.did_sell_d = self.did_buy_ba = self.did_sell_ba = self.did_buy_u = self.did_sell_u = False

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
        signal = 1.875 * c_weighted_price - p_weighted_price - 18
        u_weighted_price = self.get_weight_price(state.order_depths['UKULELE'])
        d_weighted_price = self.get_weight_price(state.order_depths['DIP'])
        pb_weighted_price = self.get_weight_price(state.order_depths['PICNIC_BASKET'])
        b_weighted_price = self.get_weight_price(state.order_depths['BAGUETTE'])
        signal2 = pb_weighted_price - 4 * d_weighted_price - 2 * b_weighted_price - u_weighted_price - 360
        pb_pos = 0
        uku = state.market_trades['UKULELE']
        for i in uku:
            if i.buyer == 'Olivia':
                self.olivia_pair = 1
            if i.seller == 'Olivia':
                self.olivia_pair = -1
        if 'PICNIC_BASKET' in state.position.keys():
            pb_pos = state.position['PICNIC_BASKET']

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
                long = self.sma(int(self.T * 0.0008), product) #####
                short = self.sma(int(self.T * 0.0002), product)
                b, a = self.find_bids(acceptable_price, order_depth, 1, 3)
                bid = max(order_depth.buy_orders.keys())
                bid = min(order_depth.sell_orders.keys())
                tilt = (curr_pos/250) * 2
                market = state.market_trades[product]
                for i in market:
                    if i.buyer == 'Olivia':
                        if self.olivia_b < 1:
                            self.olivia_b = 1
                    if i.seller == 'Olivia':
                        if self.olivia_b > -1:
                            self.olivia_b = -1
                if self.olivia_b == 1 and not self.did_buy_b: #buying from the low
                    if curr_pos == self.berries_limit:
                        self.did_buy_b = True
                        self.did_buy_at_b = t
                    orders.append(Order(product, acceptable_price + 1, (self.berries_limit-curr_pos)/2)) # -2
                    orders.append(Order(product, acceptable_price + 2, 4*(self.berries_limit-curr_pos)/10)) # 0
                    orders.append(Order(product, acceptable_price + 3, (self.berries_limit-curr_pos)/10)) # 2
                    did_buy = self.berries_limit-curr_pos
                if self.olivia_b == -1 and not self.did_sell_b: # selling from the high
                    if curr_pos == -self.berries_limit:
                        self.did_sell_b = True
                        self.didsell_at_b = t
                    orders.append(Order(product, acceptable_price - 1, (-self.berries_limit-curr_pos)/2))
                    orders.append(Order(product, acceptable_price - 2, 4*(-self.berries_limit-curr_pos)/10))
                    orders.append(Order(product, acceptable_price - 3, (-self.berries_limit-curr_pos)/10))
                    did_sell =-self.berries_limit-curr_pos
                if t >= self.T * 0.3 and t < self.T * 0.45 and did_sell == 0 and did_buy == 0 and self.did_sell_at_b < 0.35 * self.T:
                        orders.append(Order(product, acceptable_price + 1, (self.berries_limit-curr_pos)/2))
                        orders.append(Order(product, acceptable_price + 2, 4*(self.berries_limit-curr_pos)/10))
                        orders.append(Order(product, acceptable_price + 3, (self.berries_limit-curr_pos)/10))

                if t >= self.T * 0.5 and t < 0.7 * self.T and did_sell == 0 and did_buy == 0:
                    if t >= self.T * 0.55:
                        orders.append(Order(product, acceptable_price - 1, (-self.berries_limit-curr_pos)/2))
                        orders.append(Order(product, acceptable_price - 2, 4*(-self.berries_limit-curr_pos)/10))
                        orders.append(Order(product, acceptable_price - 3, (-self.berries_limit-curr_pos)/10))
                    
                if t >= 0.7 * self.T and did_sell == 0 and did_buy == 0:
                    if not self.did_sell_b:
                        orders.append(Order(product, acceptable_price + 1, (self.berries_limit-curr_pos)/2))
                        orders.append(Order(product, acceptable_price + 2, 4*(self.berries_limit-curr_pos)/10))
                        orders.append(Order(product, acceptable_price + 3, (self.berries_limit-curr_pos)/10))
                    if self.did_buy_b and self.did_sell_b:
                        if curr_pos > 0 and short[1] > long[1]:
                            orders.append(Order(product, acceptable_price - 3, -curr_pos))
                        if curr_pos < 0 and short[1] < long[1]:
                            orders.append(Order(product, acceptable_price + 3, -curr_pos))

                if t < self.T * 0.3 and self.did_buy_at_b == 0 and self.did_sell_at_b == 0:
                    if did_buy + curr_pos < self.berries_limit:
                        orders.append(Order(product, b + max(0, tilt), self.berries_limit-curr_pos-did_buy))
                    if did_sell + curr_pos > -self.berries_limit:
                        orders.append(Order(product, a - min(0, tilt), -self.berries_limit-curr_pos-did_sell))
                if t > self.T * 0.8 and self.did_buy_b and self.did_sell_b:
                    if did_buy + curr_pos < self.berries_limit:
                        orders.append(Order(product, b + max(0, tilt), self.berries_limit-curr_pos-did_buy))
                    if did_sell + curr_pos > -self.berries_limit:
                        orders.append(Order(product, a - min(0, tilt), -self.berries_limit-curr_pos-did_sell))
                result[product] = orders

            if product == 'PEARLS':
                did_buy = did_sell = 0
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                acceptable_price = 10000

                if len(order_depth.sell_orders) > 0:
                    asks = list(order_depth.sell_orders.keys())
                    asks.sort()
                    for i in asks:
                        if i <= acceptable_price:
                            ask_vol = min(-order_depth.sell_orders[i], self.pearls_limit - curr_pos - did_buy)
                            if ask_vol > 0:
                                orders.append(Order(product, i, ask_vol)) # postitive volume
                                did_buy += ask_vol

                if len(order_depth.buy_orders) > 0:
                    bids = list(order_depth.buy_orders.keys())
                    bids.sort(reverse = True)
                    for i in bids:
                        if i >= acceptable_price:
                            bid_vol = max(-order_depth.buy_orders[i], -self.pearls_limit - curr_pos - did_sell)
                            if bid_vol < 0:
                                orders.append(Order(product, i, bid_vol)) # postitive volume
                                did_sell += bid_vol
                n = 3 if order_depth.buy_orders[bids[0]] > 12 else 4
                m = 3 if order_depth.sell_orders[asks[0]] > 12 else 4
                if did_buy + curr_pos < self.pearls_limit:
                    orders.append(Order(product, acceptable_price - n, self.pearls_limit-curr_pos-did_buy))
                if did_sell + curr_pos > -self.pearls_limit:
                    orders.append(Order(product, acceptable_price + m, -self.pearls_limit-curr_pos-did_sell))
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

            if product == 'COCONUTS':  ######## TODO, need to 1. fiddle around with the range as well as the equation (quadratic?, linear?) and get ladder for bids and asks
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                weighted_price = self.get_weight_price(order_depth)
                if signal > 10:
                    orders.append(Order(product, weighted_price - 1, max(-self.coconuts_limit, -round(signal * 6))-curr_pos))
                if signal < 1 and curr_pos < 0:
                    orders.append(Order(product, weighted_price + 1, -curr_pos))
                if signal < -10:
                    orders.append(Order(product, weighted_price + 1, min(self.coconuts_limit, -round(signal * 6))-curr_pos))
                if signal > -1 and curr_pos > 0:
                    orders.append(Order(product, weighted_price - 1, -curr_pos))
                result[product] = orders

            if product == 'PINA_COLADAS':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                weighted_price = self.get_weight_price(order_depth)
                bids = list(order_depth.buy_orders.keys())
                bids.sort(reverse = True)
                asks = list(order_depth.sell_orders.keys())
                asks.sort()
                if signal > 10:
                    orders.append(Order(product, weighted_price + 2, min(self.pina_colada_limit, round(signal * 3.2))-curr_pos))
                if signal < 1 and curr_pos > 0:
                    orders.append(Order(product, weighted_price - 2, -curr_pos)) 
                if signal < -10:
                    orders.append(Order(product, weighted_price - 2, max(-self.pina_colada_limit, round(signal * 3.2))-curr_pos))
                if signal > -1 and curr_pos < 0:
                    orders.append(Order(product, weighted_price + 2, -curr_pos))
                result[product] = orders
            
            if product == 'DIVING_GEAR':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                obs = state.observations['DOLPHIN_SIGHTINGS']
                self.seen.append(obs)
                acceptable_price = self.get_weight_price(order_depth)
                self.past_data[product]['mid_price'].append(acceptable_price)
                asks = list(order_depth.sell_orders.keys())
                asks.sort()
                bids = list(order_depth.buy_orders.keys())
                bids.sort(reverse = True)
                did_buy = did_sell = 0
                long = self.sma(150, product)
                short = self.sma(50, product)

                if len(self.seen) > 1 and abs(self.seen[-1] - self.seen[-2]) > 4:
                    self.timeevent = state.timestamp + 50000 #max(100, 33 * (self.seen[-1] - self.seen[-2]))
                    self.short_price = short[1]
                    self.long_price = long[1]
                    if self.seen[-1] - self.seen[-2] > 0:
                        self.buying = True
                        self.selling = False
                        for i in asks:
                            ask_vol = min(-order_depth.sell_orders[i], self.gear_limit - curr_pos - did_buy)
                            print(ask_vol)
                            if ask_vol > 0:
                                orders.append(Order(product, i, ask_vol))
                                did_buy += ask_vol
                        if self.gear_limit - did_buy - curr_pos > 0:
                            print("BUY", str(self.gear_limit - did_buy - curr_pos) + "x", i)
                            orders.append(Order(product, asks[-1], self.gear_limit - did_buy - curr_pos))
                    else:
                        self.selling = True
                        self.buying = False
                        for i in bids:
                            bid_vol = max(-order_depth.buy_orders[i], -self.gear_limit - curr_pos - did_sell)
                            if bid_vol > 0:
                                orders.append(Order(product, i, bid_vol))
                                did_sell += bid_vol
                        if -self.gear_limit - did_sell -curr_pos < 0:
                            orders.append(Order(product, bids[-1], -self.gear_limit - curr_pos - did_sell))
                result[product] = orders
                if self.buying and curr_pos == 0 and did_buy == 0:
                    self.buying = False
                if self.selling and curr_pos == 0 and did_sell == 0:
                    self.selling = False
                if self.buying:
                    if state.timestamp > self.timeevent:
                        if long[1] >= short[1] and long[1] > self.long_price and curr_pos > 0:
                            did_sell2 = 0
                            for i in bids:
                                bid_vol = max(-order_depth.buy_orders[i], -curr_pos - did_sell2) #FUNCTION THESE!!!
                                if bid_vol > 0:
                                    orders.append(Order(product, i, bid_vol))
                                    did_sell2 += bid_vol
                            if -did_sell2 - curr_pos < 0:
                                orders.append(Order(product, bids[-1], -did_sell2 - curr_pos))
                    else:
                        if curr_pos < self.gear_limit and did_buy == 0:
                            orders.append(Order(product, acceptable_price + 2, self.gear_limit - curr_pos))
                if self.selling:
                    if state.timestamp > self.timeevent:
                        if long[1] <= short[1] and self.long_price > long[1] and curr_pos < 0:
                            did_buy2 = 0
                            for i in asks:
                                ask_vol = min(-order_depth.sell_orders[i], - curr_pos - did_buy2)
                                if ask_vol > 0:
                                    orders.append(Order(product, i, ask_vol))
                                    did_buy2 += ask_vol
                            if -did_buy2 - curr_pos < 0:
                                orders.append(Order(product, asks[-1], -did_buy2 - curr_pos))
                    else:
                        if curr_pos > -self.gear_limit and did_sell == 0:
                            orders.append(Order(product, acceptable_price - 2, -self.gear_limit - curr_pos))
                result[product] = orders

            if product == 'DIP':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                weighted_price = self.get_weight_price(order_depth)
                bids = list(order_depth.buy_orders.keys())
                bids.sort(reverse = True)
                asks = list(order_depth.sell_orders.keys())
                asks.sort()
                c = 25
                if self.olivia_pair == 1 and not self.did_buy_d:
                    if curr_pos == -self.dip_limit:
                        self.did_buy_d = True
                    else:
                        orders.append(Order(product, weighted_price, -self.dip_limit - curr_pos)) # YOLO
                if self.olivia_pair == -1 and not self.did_sell_d:
                    if curr_pos == self.berries_limit:
                        self.did_sell_d = True
                    else:
                        orders.append(Order(product, weighted_price, self.dip_limit - curr_pos)) # YOLO                
                if self.olivia_pair == 0:
                    if -4 * pb_pos - curr_pos > 0:
                        price = asks[0]
                        a, b = -1, 0
                    else:
                        price = bids[0]
                        a, b = 1, 0
                    orders.append(Order(product, price, -4 * pb_pos - curr_pos))
                result[product] = orders

            if product == 'BAGUETTE':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                weighted_price = self.get_weight_price(order_depth)
                bids = list(order_depth.buy_orders.keys())
                bids.sort(reverse = True)
                asks = list(order_depth.sell_orders.keys())
                asks.sort()
                c = 25
                if self.olivia_pair == 1 and not self.did_buy_ba:
                    if curr_pos == -self.baguette_limit:
                        self.did_buy_ba = True
                    else:
                        orders.append(Order(product, weighted_price, -self.baguette_limit - curr_pos)) # YOLO
                if self.olivia_pair == -1 and not self.did_sell_ba:
                    if curr_pos == self.berries_limit:
                        self.did_sell_ba = True
                    else:
                        orders.append(Order(product, weighted_price, self.baguette_limit - curr_pos)) # YOLO                
                if self.olivia_pair == 0:
                    if -2 * pb_pos - curr_pos > 0:
                        price = asks[0]
                        a, b = -1, 0
                    else:
                        price = bids[0]
                        a, b = 1, 0
                    orders.append(Order(product, price, -2 * pb_pos - curr_pos))
                result[product] = orders

            if product == 'UKULELE':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                weighted_price = self.get_weight_price(order_depth)
                bids = list(order_depth.buy_orders.keys())
                bids.sort(reverse = True)
                asks = list(order_depth.sell_orders.keys())
                asks.sort()
                c = 25
                if self.olivia_pair == 1 and not self.did_buy_u:
                    if curr_pos == self.ukulele_limit:
                        self.did_buy_u = True
                    else:
                        orders.append(Order(product, weighted_price, self.ukulele_limit - curr_pos)) # YOLO
                if self.olivia_pair == -1 and not self.did_sell_u:
                    if curr_pos == -self.ukulele_limit:
                        self.did_sell_u = True
                    else:
                        orders.append(Order(product, weighted_price, -self.ukulele_limit - curr_pos)) # YOLO                
                if self.olivia_pair == 0:
                    if - pb_pos - curr_pos > 0:
                        price = asks[0]
                        a, b = -1, 0
                    else:
                        price = bids[0]
                        a, b = 1, 0
                    orders.append(Order(product, price, -pb_pos - curr_pos))
                result[product] = orders

            if product == 'PICNIC_BASKET':
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                weighted_price = self.get_weight_price(order_depth)
                bids = list(order_depth.buy_orders.keys())
                bids.sort(reverse = True)
                asks = list(order_depth.sell_orders.keys())
                asks.sort()
                did_buy = did_sell = 0
                c = 25
                if self.olivia_pair == 1 and not self.did_buy_p:
                    if curr_pos == self.picnic_limit:
                        self.did_buy_p = True
                    else:
                        orders.append(Order(product, weighted_price, self.picnic_limit - curr_pos)) # YOLO
                if self.olivia_pair == -1 and not self.did_sell_p:
                    if curr_pos == -self.picnic_limit:
                        self.did_sell_p = True
                    else:
                        orders.append(Order(product, weighted_price, -self.picnic_limit - curr_pos)) # YOLO                
                if self.olivia_pair == 0:
                    if signal2 > 25:
                        for i in bids:
                            if i <= weighted_price - 8:
                                bid_vol = max(-order_depth.buy_orders[i], max(-self.picnic_limit, -round(4.69*np.sqrt(signal2 - c))) - curr_pos - did_sell)
                                if bid_vol < 0:
                                    orders.append(Order(product, i, bid_vol)) # postitive volume
                                    did_sell += bid_vol
                    if signal2 < 1 and curr_pos > 0:
                        for i in asks:
                            if i >= weighted_price + 8:
                                ask_vol = min(-order_depth.sell_orders[i], - curr_pos - did_buy)
                                if ask_vol > 0:
                                    orders.append(Order(product, i, ask_vol)) # postitive volume
                                    did_buy += ask_vol
                    if signal2 < -25:
                        for i in asks:
                            if i >= weighted_price + 8:
                                ask_vol = min(-order_depth.sell_orders[i], min(self.picnic_limit, round(4.69*np.sqrt(-signal2 - c))) - curr_pos - did_buy)
                                if ask_vol > 0:
                                    orders.append(Order(product, i, ask_vol)) # postitive volume
                                    did_buy += ask_vol
                    if signal2 > -1 and curr_pos < 0:
                        for i in bids:
                            if i <= weighted_price - 8:
                                bid_vol = max(-order_depth.buy_orders[i], - curr_pos - did_sell)
                                if bid_vol < 0:
                                    orders.append(Order(product, i, bid_vol)) # postitive volume
                                    did_sell += bid_vol
                result[product] = orders

        self.logger.flush(state, result)
        return result
    
    def prop_orders(self, orders, product, price1, price2, prop, vol):

        vol1 = math.floor(prop * vol)
        orders.append(Order(product, price1, vol1 - 1))
        orders.append(Order(product, price2, vol - vol1))