# -*- coding: utf-8 -*-

import seaborn as sns
import os
import datetime
from agents.pg import PG
from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt
from scipy import interpolate
plt.style.use('ggplot')
sns.set_style("darkgrid")

eps = 10e-8
epochs = 0
M = 0
PATH_prefix = ''


class StockTrader():
    def __init__(self):
        self.reset()

    def reset(self):
        self.wealth = 10e3
        self.total_reward = 0
        self.ep_ave_max_q = 0
        self.loss = 0
        self.actor_loss = 0

        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M))

    def update_summary(self, loss, r, q_value, actor_loss, w, p):
        self.loss += loss
        self.actor_loss += actor_loss
        self.total_reward += r
        self.ep_ave_max_q += q_value
        self.r_history.append(r)
        self.wealth = self.wealth * math.exp(r)
        self.wealth_history.append(self.wealth)
        self.w_history.extend([','.join(
            [str(Decimal(str(w0)).quantize(Decimal('0.00'))) for w0 in w.tolist()[0]])])
        self.p_history.extend([','.join(
            [str(Decimal(str(p0)).quantize(Decimal('0.000'))) for p0 in p.tolist()])])

    def write(self, codes, agent):
        global PATH_prefix
        wealth_history = pd.Series(self.wealth_history)
        r_history = pd.Series(self.r_history)
        w_history = pd.Series(self.w_history)
        p_history = pd.Series(self.p_history)
        history = pd.concat(
            [wealth_history, r_history, w_history, p_history], axis=1)
        # history.to_csv(PATH_prefix + agent + '-'.join(codes) + '-' +
        #                str(math.exp(np.sum(self.r_history)) * 100) + '.csv')
        history.to_csv(PATH_prefix + agent + '.csv')

    def print_result(self, epoch, agent, noise_flag):
        self.total_reward = math.exp(self.total_reward) * 100
        print(
            '*-----Episode: {:d}, Reward:{:.6f}%-----*'.format(epoch, self.total_reward))
        agent.write_summary(self.total_reward)
        agent.save_model()

    def plot_result(self):
        pd.Series(self.wealth_history).plot()
        plt.show()

    def action_processor(self, a, ratio):
        a = np.clip(a + self.noise() * ratio, 0, 1)
        a = a / (a.sum() + eps)
        return a


def parse_info(info):
    return info['reward'], info['continue'], info['next state'], info['weight vector'], info['price'], info['risk']


def traversal(
        stocktrader,
        agent,
        env,
        epoch,
        noise_flag,
        framework,
        method,
        trainable):
    info = env.step(None, None, noise_flag)
    r, contin, s, w1, p, risk = parse_info(info)
    contin = 1
    t = 0

    while contin:
        w2 = agent.predict(s, w1)

        env_info = env.step(w1, w2, noise_flag)
        r, contin, s_next, w1, p, risk = parse_info(env_info)

        if framework == 'PG':
            agent.save_transition(s, p, w2, w1)
        else:
            agent.save_transition(s, w2, r - risk, contin, s_next, w1)
        loss, q_value, actor_loss = 0, 0, 0

        if framework == 'DDPG':
            if not contin and trainable == "True":
                agent_info = agent.train(method, epoch)
                loss, q_value = agent_info["critic_loss"], agent_info["q_value"]
                if method == 'model_based':
                    actor_loss = agent_info["actor_loss"]

        elif framework == 'PPO':
            if not contin and trainable == "True":
                agent_info = agent.train(method, epoch)
                loss, q_value = agent_info["critic_loss"], agent_info["q_value"]
                if method == 'model_based':
                    actor_loss = agent_info["actor_loss"]

        elif framework == 'PG':
            if not contin and trainable == "True":
                agent.train()

        stocktrader.update_summary(loss, r, q_value, actor_loss, w2, p)
        s = s_next
        t = t + 1


def maxdrawdown(arr):
    i = np.argmax((np.maximum.accumulate(arr) - arr) /
                  np.maximum.accumulate(arr))  # end of the period
    j = np.argmax(arr[:i])  # start of period
    return (1 - arr[i] / arr[j])


def backtest(agent, env):
    global PATH_prefix
    print("starting to backtest......")
    from agents.UCRP import UCRP
    from agents.Winner import WINNER
    from agents.Losser import LOSSER

    agents = []
    agents.extend(agent)
    agents.append(WINNER())
    agents.append(UCRP())
    agents.append(LOSSER())
    labels = ['PG', 'Winner', 'UCRP', 'Losser']
    wealths_result = []
    rs_result = []
    for i, agent in enumerate(agents):
        stocktrader = StockTrader()
        info = env.step(None, None, 'False')
        r, contin, s, w1, p, risk = parse_info(info)
        contin = 1
        wealth = 10000
        wealths = [wealth]
        rs = [1]
        while contin:
            w2 = agent.predict(s, w1)
            env_info = env.step(w1, w2, 'False')
            r, contin, s_next, w1, p, risk = parse_info(env_info)
            wealth = wealth * math.exp(r)
            rs.append(math.exp(r) - 1)
            wealths.append(wealth)
            s = s_next
            stocktrader.update_summary(0, r, 0, 0, w2, p)

        stocktrader.write(map(lambda x: str(x), env.get_codes()), labels[i])
        print('finish one agent')
        wealths_result.append(wealths)
        rs_result.append(rs)

    print('资产名称', '   ', '平均日收益率', '   ', '夏普率', '   ', '最大回撤')
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(len(agents)):
        # if labels[i] == 'UCRP' or labels[i] == 'Losser':
        #     continue
        plt.plot(wealths_result[i], label=labels[i])
        fileObject = open(labels[i] + '.txt', 'w')
        for ip in wealths_result[i]:
            fileObject.write(str(ip))
            fileObject.write(', ')
        fileObject.close()
        mrr = float(np.mean(rs_result[i]) * 100)
        sharpe = float(
            np.mean(
                rs_result[i]) /
            np.std(
                rs_result[i]) *
            np.sqrt(252))
        maxdrawdown = float(max(1 -
                                min(wealths_result[i]) /
                                np.maximum.accumulate(wealths_result[i])))
        print(
            labels[i], '   ', round(
                mrr, 3), '%', '   ', round(
                sharpe, 3), '  ', round(
                maxdrawdown, 3))
    # new_data = []
    # for i in LSTM:
    #     for j in range(8):
    #         new_data.append(i)
    # inter_op = interpolate.interp1d(range(len(hand_data)), hand_data, kind='linear')
    # hand_data = inter_op(7*range(len(hand_data)))

    # def longer(i):
    #     k = np.zeros(180)
    #     j = 0
    #     p = 0
    #     while j + 1 < len(i):
    #         print([p, j])
    #         k[p] = i[j]
    #         k[p + 1] = i[j]
    #         k[p + 2] = i[j + 1]
    #         p += 3
    #         j += 2
    #     return i
    # LSTM = longer(LSTM)
    # Random = longer(Random)
    # Uniform = longer(Uniform)
    # plt.plot(LSTM, label='LSTM')
    # plt.plot(Random, label='Random')
    # plt.plot(Uniform, label='Uniform')
    plt.legend()
    plt.savefig(PATH_prefix + 'backtest_with_hand.png')
    plt.show()


def parse_config(config, mode):
    codes = config["session"]["codes"]
    start_date = config["session"]["start_date"]
    end_date = config["session"]["end_date"]
    features = config["session"]["features"]
    agent_config = config["session"]["agents"]
    market = config["session"]["market_types"]
    noise_flag, record_flag, plot_flag = config["session"]["noise_flag"], config[
        "session"]["record_flag"], config["session"]["plot_flag"]
    predictor, framework, window_length = agent_config
    reload_flag, trainable = config["session"]['reload_flag'], config["session"]['trainable']
    method = config["session"]['method']

    global epochs
    epochs = int(config["session"]["epochs"])

    if mode == 'test':
        record_flag = 'True'
        noise_flag = 'False'
        plot_flag = 'True'
        reload_flag = 'True'
        trainable = 'False'
        method = 'model_free'

    print("*--------------------Training Status-------------------*")
    print("Date from", start_date, ' to ', end_date)
    print('Features:', features)
    print(
        "Agent:Noise(",
        noise_flag,
        ')---Recoed(',
        noise_flag,
        ')---Plot(',
        plot_flag,
        ')')
    print("Market Type:", market)
    print(
        "Predictor:",
        predictor,
        "  Framework:",
        framework,
        "  Window_length:",
        window_length)
    print("Epochs:", epochs)
    print("Trainable:", trainable)
    print("Reloaded Model:", reload_flag)
    print("Method", method)
    print("Noise_flag", noise_flag)
    print("Record_flag", record_flag)
    print("Plot_flag", plot_flag)

    return codes, start_date, end_date, features, agent_config, market, predictor, framework, window_length, noise_flag, record_flag, plot_flag, reload_flag, trainable, method


def session(config, args):
    global PATH_prefix
    from data.environment import Environment
    codes, start_date, end_date, features, agent_config, market, predictor, framework, window_length, noise_flag, record_flag, plot_flag, reload_flag, trainable, method = parse_config(
        config, args)
    env = Environment()

    global M
    M = codes + 1

    # if framework == 'DDPG':
    #     print("*-----------------Loading DDPG Agent---------------------*")
    #     from agents.ddpg import DDPG
    #     agent = DDPG(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,trainable)
    #
    # elif framework == 'PPO':
    #     print("*-----------------Loading PPO Agent---------------------*")
    #     from agents.ppo import PPO
    #     agent = PPO(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,trainable)

    stocktrader = StockTrader()
    PATH_prefix = "result/PG/" + str(args['num']) + '/'

    if args['mode'] == 'train':
        if not os.path.exists(PATH_prefix):
            os.makedirs(PATH_prefix)
            train_start_date, train_end_date, test_start_date, test_end_date, codes = env.get_repo(
                start_date, end_date, codes, market)
            env.get_data(
                train_start_date,
                train_end_date,
                features,
                window_length,
                market,
                codes)
            print("Codes:", codes)
            print(
                'Training Time Period:',
                train_start_date,
                '   ',
                train_end_date)
            print(
                'Testing Time Period:',
                test_start_date,
                '   ',
                test_end_date)
            with open(PATH_prefix + 'config.json', 'w') as f:
                json.dump({"train_start_date": train_start_date.strftime('%Y-%m-%d'),
                           "train_end_date": train_end_date.strftime('%Y-%m-%d'),
                           "test_start_date": test_start_date.strftime('%Y-%m-%d'),
                           "test_end_date": test_end_date.strftime('%Y-%m-%d'), "codes": codes}, f)
                print("finish writing config")
        else:
            with open("result/PG/" + str(args['num']) + '/config.json', 'r') as f:
                dict_data = json.load(f)
                print("successfully load config")
            train_start_date, train_end_date, codes = datetime.datetime.strptime(
                dict_data['train_start_date'], '%Y-%m-%d'), datetime.datetime.strptime(
                dict_data['train_end_date'], '%Y-%m-%d'), dict_data['codes']
            env.get_data(
                train_start_date,
                train_end_date,
                features,
                window_length,
                market,
                codes)

        for noise_flag in [
                'True']:  # ['False','True'] to train agents with noise and without noise in assets prices
            if framework == 'PG':
                print("*-----------------Loading PG Agent---------------------*")
                agent = PG(
                    len(codes) + 1,
                    int(window_length),
                    len(features),
                    '-'.join(agent_config),
                    reload_flag,
                    trainable,
                    noise_flag,
                    args['num'])

            print("Training with {:d} Epochs".format(epochs))
            for epoch in range(epochs):
                print("Now we are at epoch", epoch)
                traversal(
                    stocktrader,
                    agent,
                    env,
                    epoch,
                    noise_flag,
                    framework,
                    method,
                    trainable)

                if record_flag == 'True':
                    stocktrader.write(epoch, framework)

                if plot_flag == 'True':
                    stocktrader.plot_result()

                agent.reset_buffer()
                stocktrader.print_result(epoch, agent, noise_flag)
                stocktrader.reset()
            agent.close()
            del agent

    elif args['mode'] == 'test':
        with open("result/PG/" + str(args['num']) + '/config.json', 'r') as f:
            dict_data = json.load(f)
        test_start_date, test_end_date, codes = datetime.datetime.strptime(
            dict_data['test_start_date'], '%Y-%m-%d'), datetime.datetime.strptime(
            dict_data['test_end_date'], '%Y-%m-%d'), dict_data['codes']
        env.get_data(
            test_start_date,
            test_end_date,
            features,
            window_length,
            market,
            codes)
        backtest([PG(len(codes) + 1,
                     int(window_length),
                     len(features),
                     '-'.join(agent_config),
                     'True',
                     'False',
                     'True',
                     args['num'])],
                 env)


def build_parser():
    parser = ArgumentParser(
        description='Provide arguments for training different DDPG or PPO models in Portfolio Management')
    parser.add_argument("--mode", choices=['train', 'test', 'download'])
    parser.add_argument("--num", type=int)
    return parser


def main():
    parser = build_parser()
    args = vars(parser.parse_args())
    with open('config.json') as f:
        config = json.load(f)
        if args['mode'] == 'download':
            from data.download_data import DataDownloader
            data_downloader = DataDownloader(config)
            data_downloader.save_data()
        else:
            session(config, args)


if __name__ == "__main__":
    main()
