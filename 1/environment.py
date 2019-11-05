# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 05:41:28 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
from math import log
from datetime import datetime
import time
import random

eps = 10e-8


def fill_zeros(x):
    return '0' * (6 - len(x)) + x


class Environment:
    def __init__(self):
        self.cost = 0.0025

    def get_repo(self, start_date, end_date, codes_num, market):
        # preprocess parameters

        # read all data
        self.data = pd.read_csv(
            r'./data/' + market + '.csv',
            index_col=0,
            parse_dates=True,
            dtype=object)
        self.data["code"] = self.data["code"].astype(str)
        if market == 'China':
            self.data["code"] = self.data["code"].apply(fill_zeros)

        sample_flag = True
        while sample_flag:
            codes = random.sample(set(self.data["code"]), codes_num)
            data2 = self.data.loc[self.data["code"].isin(codes)]

            # 生成有效时间
            date_set = set(data2.loc[data2['code'] == codes[0]].index)
            for code in codes:
                date_set = date_set.intersection(
                    (set(data2.loc[data2['code'] == code].index)))
            if market == 'America':
                #if len(date_set) > 2000:
                sample_flag = False
            elif market == "China":
                #if len(date_set) > 2400:
                    sample_flag = False


        date_set = date_set.intersection(
            set(pd.date_range(start_date, end_date)))
        self.date_set = list(date_set)
        self.date_set.sort()

        train_start_time = self.date_set[0]
        train_end_time = self.date_set[int(len(self.date_set) / 6) * 5 - 1]
        test_start_time = self.date_set[int(len(self.date_set) / 6) * 5]
        test_end_time = self.date_set[-1]

        return train_start_time, train_end_time, test_start_time, test_end_time, codes

    def get_data(
            self,
            start_time,
            end_time,
            features,
            window_length,
            market,
            codes):
        self.codes = codes

        self.data = pd.read_csv(
            r'./data/' + market + '.csv',
            index_col=0,
            parse_dates=True,
            dtype=object)
        self.data["code"] = self.data["code"].astype(str)
        if market == 'China':
            self.data["code"] = self.data["code"].apply(fill_zeros)

        self.data[features] = self.data[features].astype(float)
        self.data = self.data[start_time.strftime(
            "%Y-%m-%d"):end_time.strftime("%Y-%m-%d")]
        data = self.data
        # TO DO:REFINE YOUR DATA

        # Initialize parameters
        self.M = len(codes) + 1
        self.N = len(features)
        self.L = int(window_length)
        self.date_set = pd.date_range(start_time, end_time)
        # 为每一个资产生成数据
        asset_dict = dict()  # 每一个资产的数据
        for asset in codes:
            # 加入时间的并集，会产生缺失值pd.to_datetime(self.date_list)
            asset_data = data[data["code"] == asset].reindex(
                self.date_set).sort_index()
            asset_data = asset_data.resample('D').mean()
            asset_data['close'] = asset_data['close'].fillna(method='pad')
            base_price = asset_data.ix[-1, 'close']
            asset_dict[str(asset)] = asset_data
            asset_dict[str(asset)]['close'] = asset_dict[str(
                asset)]['close'] / base_price

            if 'high' in features:
                asset_dict[str(asset)]['high'] = asset_dict[str(
                    asset)]['high'] / base_price

            if 'low' in features:
                asset_dict[str(asset)]['low'] = asset_dict[str(
                    asset)]['low'] / base_price

            if 'open' in features:
                asset_dict[str(asset)]['open'] = asset_dict[str(
                    asset)]['open'] / base_price

            asset_data = asset_data.fillna(method='bfill', axis=1)
            asset_data = asset_data.fillna(
                method='ffill', axis=1)  # 根据收盘价填充其他值
            #***********************open as preclose*******************#
            # asset_data=asset_data.dropna(axis=0,how='any')
            asset_dict[str(asset)] = asset_data

        # 开始生成tensor
        self.states = []
        self.price_history = []
        t = self.L + 1
        length = len(self.date_set)
        while t < length - 1:
            V_close = np.ones(self.L)
            if 'high' in features:
                V_high = np.ones(self.L)
            if 'open' in features:
                V_open = np.ones(self.L)
            if 'low' in features:
                V_low = np.ones(self.L)

            y = np.ones(1)
            state = []
            for asset in codes:
                asset_data = asset_dict[str(asset)]
                V_close = np.vstack(
                    (V_close, asset_data.ix[t - self.L - 1:t - 1, 'close']))
                if 'high' in features:
                    V_high = np.vstack(
                        (V_high, asset_data.ix[t - self.L - 1:t - 1, 'high']))
                if 'low' in features:
                    V_low = np.vstack(
                        (V_low, asset_data.ix[t - self.L - 1:t - 1, 'low']))
                if 'open' in features:
                    V_open = np.vstack(
                        (V_open, asset_data.ix[t - self.L - 1:t - 1, 'open']))
                y = np.vstack(
                    (y, asset_data.ix[t, 'close'] / asset_data.ix[t - 1, 'close']))
            state.append(V_close)
            if 'high' in features:
                state.append(V_high)
            if 'low' in features:
                state.append(V_low)
            if 'open' in features:
                state = np.stack((state, V_open), axis=2)

            state = np.stack(state, axis=1)
            state = state.reshape(1, self.M, self.L, self.N)
            self.states.append(state)
            self.price_history.append(y)
            t = t + 1
        self.reset()

    def step(self, w1, w2, noise):
        if self.FLAG:
            not_terminal = 1
            price = self.price_history[self.t]
            if noise == 'True':
                price = price + \
                    np.stack(np.random.normal(0, 0.002, (1, len(price))), axis=1)
            mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()

            # std = self.states[self.t - 1][0].std(axis=0, ddof=0)
            # w2_std = (w2[0]* std).sum()

            # #adding risk
            # gamma=0.00
            # risk=gamma*w2_std

            risk = 0
            r = (np.dot(w2, price)[0] - mu)[0]

            reward = np.log(r + eps)

            w2 = w2 / (np.dot(w2, price) + eps)
            self.t += 1
            if self.t == len(self.states):
                not_terminal = 0
                self.reset()

            price = np.squeeze(price)
            info = {'reward': reward,
                    'continue': not_terminal,
                    'next state': self.states[self.t],
                    'weight vector': w2,
                    'price': price,
                    'risk': risk}
            return info
        else:
            info = {'reward': 0,
                    'continue': 1,
                    'next state': self.states[self.t],
                    'weight vector': np.array([[1] + [0 for i in range(self.M - 1)]]),
                    'price': self.price_history[self.t],
                    'risk': 0}

            self.FLAG = True
            return info

    def reset(self):
        self.t = self.L + 1
        self.FLAG = False

    def get_codes(self):
        return self.codes
