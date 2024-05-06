import pygambit as gbt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
import os
import csv
import sys
from pathlib import Path

n = int(sys.argv[1])
t = int(sys.argv[2])
stocks = int(sys.argv[3])
options = int(sys.argv[4])
spread = int(sys.argv[5])
p = np.zeros(stocks) # p[j] denotes the current market price of stock j
capital = np.zeros(n) # capital[i] denotes the current capital with player i
initial_capital = np.zeros(n) # initial_capital[i] denotes the initial capital (including value of stocks) at time t = 0
portfolio = np.zeros((n,stocks)) # portfolio[i][j] denotes the number of shares of stock j that player i has
risk = []
for i in range(n):
    risk.append(float(sys.argv[6+i]))

print('n = ' + str(n))
print('t = ' + str(t))
print('stocks = ' + str(stocks))
print('options = ' + str(options))
print('spread = ' + str(spread))
print('risk =', end = " ")
print(risk)

temp_folder_name = ""
for i in range(len(sys.argv)-1):
    temp_folder_name+=sys.argv[1+i]

# Specify the directory path
temp_directory = "/home/maths/dual/mt6200886/scratch/project/"+temp_folder_name

# Create the temp_directory
path = Path(temp_directory)
path.mkdir(parents=True, exist_ok=True)

class Strategy:
    def __init__(self, price = 0, action = "hold"):
        self.price = price
        self.action = action

class Bid_Ask_List:
    def __init__(self, price = 0, player = -1):
        self.price = price
        self.player = player
    
    def __eq__(self, other):
        return self.price == other.price
    
    def __lt__(self, other):
        return self.price < other.price

#strategy[i][j][k][l] is a tuple of (hypothesized price, action) for 'i'th player at time epoch 'j' for the 'k'th stock and option chosen is 'l'
strategy = [[[[Strategy() for l in range(options)] for k in range(stocks)] for j in range(t)] for i in range(n)]

#current_strategy[i][j][k] is the chosen strategies combination , a tuple of (hypothesized price, action) for 'i'th player at time epoch 'j' for the 'k'th stock
current_strategy = [[[Strategy() for k in range(stocks)] for j in range(t)] for i in range(n)]

data = []
directory = '/home/maths/dual/mt6200886/project/yahoo_data_' + str(stocks)
 
for filename in os.listdir(directory):
    temp_data = []
    with open(directory + '/' + filename, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            date = row[0]
            close_price = float(row[1])
            temp_data.append((date, close_price))

    series_data = pd.Series([x[1] for x in temp_data], index=[x[0] for x in temp_data])
    data.append(series_data)
    
def calculate_prices(data,days,trials):
    log_return = np.log(1 + data.pct_change()) # periodic daily return = ln(today's price/previous day's price)
    u = log_return.mean() 
    var = log_return.var()
    drift = u - (0.5*var) # drift, used for direction
    stdev = log_return.std()
    Z = norm.ppf(np.random.rand(days, trials))
    daily_returns = np.exp(drift + stdev * Z)
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data.iloc[-1]
    for t in range(1, days):
        price_paths[t] = price_paths[t-1]*daily_returns[t]     
    return price_paths

def strategy_initialise():
    
    # using Monte Carlo simulation 
    days = t+1
    trials = 10000
    prices = []
    mean = []
    sigma = []
    for i in range(stocks):
        prices.append(calculate_prices(data[i],days,trials))
        mean.append(np.zeros(days))
        sigma.append(np.zeros(days))
        for j in range(days):
            mean[i][j] = np.mean(prices[i][j])
            sigma[i][j] = np.std(prices[i][j])
            
    for i in range(n):
        for j in range(t):
            for k in range(stocks):
                lower_bound = mean[k][j+1]-risk[i]*sigma[k][j+1]
                upper_bound = mean[k][j+1]+risk[i]*sigma[k][j+1]
                random_numbers = [random.uniform(lower_bound, upper_bound) for _ in range(options)]
                avg = sum(random_numbers)/len(random_numbers)
                for l in range(options):
                    strategy[i][j][k][l].price = random_numbers[l]
                    if(random_numbers[l]>avg):
                        strategy[i][j][k][l].action = "sell"
                    elif(random_numbers[l]<avg):
                        strategy[i][j][k][l].action = "buy"
                    else:
                        strategy[i][j][k][l].action = "hold"
    
    return mean

def initialise():
    for i in range(n):
        capital[i] = 500
        for j in range(stocks):
            portfolio[i][j]=2
            
    for i in range(stocks):
        p[i] = mean_price[i][0]
    
    for i in range(n):
        initial_capital[i] = capital[i]
        for j in range(stocks):
            initial_capital[i]+=p[j]*portfolio[i][j]
            
    
def check_validity(current_strategy):
    for i in range(t):
        bid_present = False
        ask_present = False
        min_difference = 100000000
        for s in range(stocks):
            highest_bid = -100000000
            lowest_ask = 100000000
            for j in range(n):
                if current_strategy[j][i][s].action=="sell":
                    ask_present = True
                    lowest_ask = min(lowest_ask,current_strategy[j][i][s].price)
                if current_strategy[j][i][s].action=="buy":
                    bid_present = True
                    highest_bid = max(highest_bid,current_strategy[j][i][s].price)
            min_difference = min(lowest_ask-highest_bid,min_difference)
        if(bid_present==False or ask_present==False):
            return False
        if(min_difference>spread):
            return False
    return True

def execute_trade(current_strategy): 
    for time in range(t):
        for j in range(stocks):
            bid_list = []
            ask_list = []
            for i in range(n):
                if(current_strategy[i][time][j].action=="sell"):
                    ask_list.append(Bid_Ask_List(current_strategy[i][time][j].price,i))
                if(current_strategy[i][time][j].action=="buy"):
                    bid_list.append(Bid_Ask_List(current_strategy[i][time][j].price,i))
            bid_list.sort()
            bid_list.reverse()
            ask_list.sort()
            match_index = -1
            execution_price = 0
            for i in range(min(len(bid_list),len(ask_list))):
                if(ask_list[i].price-bid_list[i].price<=spread):
                    match_index = i
                    avg_bid_ask = (ask_list[i].price+bid_list[i].price)/2
                    execution_price = (i*execution_price+avg_bid_ask)/(i+1)
            # updating the market price
            p[j] = execution_price
            for i in range(min(match_index+1,n)):
                capital[ask_list[i].player]+=execution_price
                capital[bid_list[i].player]-=execution_price
                portfolio[ask_list[i].player][j]-=1
                portfolio[bid_list[i].player][j]+=1
        
    
def calculate_payoff():
    profit = np.zeros(n)
    for i in range(n):
        profit[i] = capital[i]-initial_capital[i]
        for j in range(stocks):
            profit[i]+=p[j]*portfolio[i][j]
    
    return profit
    
mean_price = strategy_initialise()
initialise()

k = n*t*stocks
for i in range(k):
    for j in range(options):
        if i==0:
            if j==0:
                with open(temp_directory + '/file0.txt', 'w') as file:
                    file.write(str(j)+"\n")
            elif j==options-1:
                with open(temp_directory + '/file0.txt', 'a') as file:
                    file.write(str(j)+"\nEOF")
            else:
                with open(temp_directory + '/file0.txt', 'a') as file:
                    file.write(str(j)+"\n")
        else:
            f = open(temp_directory + f"/file{i-1}.txt", "r")
            f2 = open(temp_directory + f"/file{i}.txt", "w")
            num = f.readline()
            while num!="EOF":
                curr = num[:-1]
                for m in range(options):
                    f2.write(curr + str(m) + '\n')
                num = f.readline()
            f2.write("EOF")
    if i>0:
        os.remove(temp_directory + f"/file{i-1}.txt")

def convert(arr):
    st = ""
    for a in arr:
        st+=str(a)
    return st

payoff_strategy = {}
valid_strategies = []
valid_strategies_str = []
f = open(temp_directory + f"/file{n*t*stocks-1}.txt", "r")
strat = f.readline()
ind = 0
while strat!="EOF":
    strat = strat[:-1]
    for opt in range(len(strat)):
        i = opt//(t*stocks)
        j = (opt%(t*stocks))//stocks
        k = ((opt%(t*stocks))%stocks)
        current_strategy[i][j][k] = strategy[i][j][k][int(strat[opt])]
    if(check_validity(current_strategy)):
        initialise()
        execute_trade(current_strategy)
        payoff_strategy[strat] = calculate_payoff()
        valid_strategies_str.append(strat)
        current_strategy = [[[Strategy() for k in range(stocks)] for j in range(t)] for i in range(n)]
    strat = f.readline()
    ind+=1

unique_strategies = [set() for i in range(n)]
for vstrategy in valid_strategies_str:
    k = len(vstrategy)//n
    for i in range(n):
        player_strategy = vstrategy[i*k:(i+1)*k]
        unique_strategies[i].add(player_strategy)
for i in range(len(unique_strategies)):
    unique_strategies[i] = list(unique_strategies[i])
    
dims = [n]
for elem in unique_strategies:
    dims.append(len(elem))
dims = tuple(dims)
player_matrices = np.zeros(dims)
for player in range(n):
    shape = player_matrices[0].shape
    for coordinate in np.ndindex(shape):
        st = ""
        for i, dim in enumerate(coordinate):
            st+=unique_strategies[i][dim]
        if st in valid_strategies_str:
            player_matrices[player][coordinate]= payoff_strategy[st][player]
        else:
            player_matrices[player][coordinate] = 0

player_matrices = player_matrices.astype(gbt.Rational)
g = gbt.Game.from_arrays(*player_matrices)
result = gbt.nash.logit_solve(g)
equilibrium_profile = result.equilibria[0]
expected_payoffs = {"Player "+str(i): float(equilibrium_profile.payoff(str(i+1))) for i in range(n)}

# Print the expected payoffs for each player
print("Expected Payoffs at Equilibrium:")
for player, payoff in expected_payoffs.items():
    print(f"{player}: {payoff}")

os.remove(temp_directory + f"/file{n*t*stocks-1}.txt")
os.rmdir(temp_directory)