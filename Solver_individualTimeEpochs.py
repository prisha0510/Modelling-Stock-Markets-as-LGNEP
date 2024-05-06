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
from classes import Strategy, Bid_Ask_List
import time

timestamp = str(time.time())
print("timestamp = " + timestamp)
n = int(sys.argv[1])
options = int(sys.argv[2])
spread = int(sys.argv[3])
iters = int(sys.argv[4])
p = np.zeros(1) # p[j] denotes the current market price of stock j
market_price = np.zeros(iters)
capital = np.zeros(n) # capital[i] denotes the current capital with player i
initial_capital = np.zeros(n) # initial_capital[i] denotes the initial capital (including value of stocks) at time t = 0
portfolio = np.zeros(n) # portfolio[i] denotes the number of shares that player i has
risk = []
for i in range(n):
    risk.append(float(sys.argv[5+i]))

print('n = ' + str(n))
print('options = ' + str(options))
print('spread = ' + str(spread))
print('risk =', end = " ")
print(risk)

temp_folder_name = timestamp
for i in range(len(sys.argv)-1):
    temp_folder_name+=sys.argv[1+i]

# Specify the directory path
temp_directory = "/home/maths/dual/mt6200886/scratch/project/"+temp_folder_name

# Create the temp_directory
path = Path(temp_directory)
path.mkdir(parents=True, exist_ok=True)

#strategy[i][k] is a tuple of (hypothesized price, action) for 'i'th player  and option chosen is 'k'
strategy = [[Strategy() for k in range(options)] for i in range(n)]

#current_strategy[i] is the chosen strategies combination , a tuple of (hypothesized price, action) for 'i'th player
current_strategy = [Strategy()  for i in range(n)]

def initialise(iter_no, mean_price):
    global capital
    global portfolio
    global p
    if iter_no==0:
        for i in range(n):
            capital[i] = 500
            portfolio[i]=2

        p[0] = mean_price[0]
    else:
        capital = np.load(temp_directory + f'/capital_{iter_no-1}.npy')
        portfolio = np.load(temp_directory + f'/portfolio_{iter_no-1}.npy')
        p = np.load(temp_directory + f'/p_{iter_no-1}.npy')
        # os.remove(f'capital_{iter_no-1}.npy')
        # os.remove(f'portfolio_{iter_no-1}.npy')
        # os.remove(f'p_{iter_no-1}.npy')
    
    for i in range(n):
        initial_capital[i] = capital[i]
        initial_capital[i]+=p[0]*portfolio[i]
            
    
def check_validity(current_strategy):
    bid_present = False
    ask_present = False
    min_difference = 100000000
    
    highest_bid = -100000000
    lowest_ask = 100000000
    for j in range(n):
        if current_strategy[j].action=="sell":
            ask_present = True
            lowest_ask = min(lowest_ask,current_strategy[j].price)
        if current_strategy[j].action=="buy":
            bid_present = True
            highest_bid = max(highest_bid,current_strategy[j].price)
    min_difference = min(lowest_ask-highest_bid,min_difference)
    if(bid_present==False or ask_present==False):
        return False
    if(min_difference>spread):
        return False
    return True

def execute_trade(current_strategy): 
    bid_list = []
    ask_list = []
    for i in range(n):
        if(current_strategy[i].action=="sell"):
            ask_list.append(Bid_Ask_List(current_strategy[i].price,i))
        if(current_strategy[i].action=="buy"):
            bid_list.append(Bid_Ask_List(current_strategy[i].price,i))
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
    p[0] = execution_price
    for i in range(min(match_index+1,n)):
        capital[ask_list[i].player]+=execution_price
        capital[bid_list[i].player]-=execution_price
        portfolio[ask_list[i].player]-=1
        portfolio[bid_list[i].player]+=1

    
def calculate_payoff():
    profit = np.zeros(n)
    for i in range(n):
        profit[i] = capital[i]-initial_capital[i]
        profit[i]+=p[0]*portfolio[i]
    
    return profit

def strategy_initialise(iter_no):
    # Load the arrays from the file
    mean, sigma = np.load('mean&sigma.npy')
    for i in range(n):
        j = iter_no
        lower_bound = mean[j+1]-risk[i]*sigma[j+1]
        upper_bound = mean[j+1]+risk[i]*sigma[j+1]
        random_numbers = [random.uniform(lower_bound, upper_bound) for _ in range(options)]
        avg = sum(random_numbers)/len(random_numbers)
        for l in range(options):
            strategy[i][l].price = random_numbers[l]
            if(random_numbers[l]>avg):
                strategy[i][l].action = "sell"
            elif(random_numbers[l]<avg):
                strategy[i][l].action = "buy"
            else:
                strategy[i][l].action = "hold"
    
    return mean

def run(iter_no):
    global n, options, spread, risk, strategy, current_strategy, p, capital, portfolio, temp_directory, expected_payoffs_final
    mean_price = strategy_initialise(iter_no)
    initialise(iter_no, mean_price)

    k = n
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

    payoff_strategy = {}
    valid_strategies = []
    valid_strategies_str = []
    f = open(temp_directory + f"/file{n-1}.txt", "r")
    strat = f.readline()
    ind = 0
    while strat!="EOF":
        strat = strat[:-1]
        for opt in range(len(strat)):
            i = opt
            current_strategy[i] = strategy[i][int(strat[opt])]
        if(check_validity(current_strategy)):
            initialise(iter_no, mean_price)
            execute_trade(current_strategy)
            payoff_strategy[strat] = calculate_payoff()
            valid_strategies_str.append(strat)
            current_strategy = [Strategy()for i in range(n)]
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
    # print("Expected Payoffs at Equilibrium:")
    for player, payoff in expected_payoffs.items():
        expected_payoffs_final[int(player[-1])][iter_no] = payoff
    #     print(f"{player}: {payoff}")

    np.save(temp_directory + f'/capital_{iter_no}.npy', capital)
    np.save(temp_directory + f'/portfolio_{iter_no}.npy', portfolio)
    market_price[iter_no] = p[0]
    np.save(temp_directory + f'/p_{iter_no}.npy', p)

expected_payoffs_final = np.zeros((n, iters))
for i in range(iters):
    run(i)

os.remove(temp_directory + f"/file{n-1}.txt")
for iter_no in range(iters):
    os.remove(temp_directory + f'/capital_{iter_no}.npy')
    os.remove(temp_directory + f'/portfolio_{iter_no}.npy')
    os.remove(temp_directory + f'/p_{iter_no}.npy')
os.rmdir(temp_directory)
print("Expected payoffs")
for i in range(n):
    print("Player" + str(i))
    print(np.round(expected_payoffs_final[i],decimals = 2).tolist())
print("Market price")
print(market_price.tolist())
print("Final total profit")
final_profit = np.zeros(n)
for i in range(n):
    final_profit[i] = capital[i]-500 + p[0]*portfolio[i]
print(final_profit.tolist())