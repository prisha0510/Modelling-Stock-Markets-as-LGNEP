# Modelling-Stock-Markets-as-LGNEP
This project aims to model the stock market as an LGNEP (Linear Generalised Nash Equilibrium Problem) and we use Game Theoretic techniques for Analysis of different players/participants in the Stock Markets
We aim to model the stock market and use the techniques of GNEPs (Generalised Nash Equilibrium Problems) to find the expected equilibrium profit and loss of all market participants. Our setup consists of different players with different ideologies; for instance, a retail investor might be interested in short-term investment(that is, his goal can be to make short-term profits), whereas corporate investment banks might have more long-term goals. Based on this, their strategies in the market can be different.
This is a dynamic game setup, so we consider the game and update variables at each time epoch t. The overall aim is to study the expected equilibrium profits/losses made by each player and analyze them at the end of some time period T .
# Assumptions
1. We consider each time epoch t to be a day, and we assume that intra-day trading and short selling are not allowed.
2. The time period T is considered to be finite.
3. We assume that the bids/asks are placed for one unit of shares (for instance, one unit might comprise 1000 shares)

# Defining the Game and Variables
Our game setting is dynamic; we consider the game at each epoch. The players in the market vary according to the level of risk they are willing to take, and the difference in their strategy sets denotes this.
## Strategy Sets
We shall define each player’s strategy sets at each time epoch t before the game begins. At the beginning, we define strategy sets for each player, consisting of a tuple denoting the player’s hypothesized price and his/her action at time t. The player can also have no action; if the player runs out of capital, we assume they can no longer play, as we are not allowing short selling.
## Generation of Strategy Sets
### Using Monte Carlo Simulations
We will select players’ strategies from a probability distribution of prices derived using Monte Carlo simulations from historical data over the past T time epochs. We take the mean (μ) as the hypothesized price and σ as the standard deviation of the distribution (and we use it as a measure of risk). The player will randomly choose the hypothesized value from (μ − kσ, μ + kσ), where k depends on the risk the player is willing to take.

### Using LSTM model
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) architecture that excels at capturing patterns in sequential data while mitigating the vanishing gradient problem often encountered in traditional
RNNs. In predicting stock prices, LSTMs are employed due to their ability to learn from historical price data and capture intricate dependencies over time. LSTMs maintain a cell state that can store information for long durations, selectively updating and forgetting information using gate mechanisms. These gates regulate the flow of information into and out of the cell state, allowing LSTMs to remember or forget past observations as needed. This capability makes them well-suited for modeling the complex, non-linear relationships in
stock price data. Historical price data is typically fed into the network as sequential input sequences to predict stock prices using LSTMs. The LSTM learns from these sequences to identify patterns and relationships between past price movements and future price trends. Training the network on historical data and optimizing its parameters can predict future stock prices based on the learned patterns. The model which we have trained consists of 4 layers.

# Execution of Trades
1. We will define a region (spread) and consider the bids and asks in this spread. Let B denote the number of bids in this region, and let A denote the number of asks in this region. Denote M = min(A, B). We will consider the lowest M asks and highest M bids for matching.
2. The price at which the order is executed at the average of M bid and M ask prices

# Objective Function
In this project, we have made 2 variations of the game problem
1. In the first approach, we solve game problem considering the utility function which takes into account only the final profit. The aim here for each participant is to maximise his/her profits in the long run. The program for this is included in Solver_FinalTimeEpoch.py
  The utility function used here is (Capital)^T - (Capital)^0 + (Unrealised Profit)^T - (Unrealised Profit)^0
3. In the second approach, we solve the game problem considering the utility function which maximises the profit of each player at each individual time epoch. The program for this is included in Solver_individualTimeEpochs.py
   The utility function used here is (Capital)^t - (Capital)^(t-1) + (Unrealised Profit)^t - (Unrealised Profit)^(t-1)
Note : Unrealised profit represents the monetary value of stocks at any given time and is calculated as Price of stock at that time multiplied with the number of stocks

# Justification for categorising the problem as LGNEP
LGNEP represents a class of problems where the players have their strategy sets dependent on each other which can arise due to a number of reasons. A shared constraint is one of them. The assumption here for any stock is that the sum of the number of shares of a particular stock with all the players at any given time is always constant. This helps us formulate the problem as an LGNEP

# Approach to Analyze the Equilibirum
We typically analyze two-player games using bimatrix representation. Extending this idea to the n-player model, we get an nk matrix. Now, we try to find the expected Mixed Nash Equilibrium payoffs and can analyze the risk vs reward ratio based on the players’ chosen strategy. As the simulation progresses, we capture data from every epoch to extract the price movement of the stocks. Finally, we analyze the payoffs and find the best strategies for a player to choose, like a support vector containing strategies that significantly contribute to mixed Nash equilibrium. We will be solving the game problem at each time epoch due to reasons of computational efficiency. For the same reasons, we will represent the market as groups of people rather than individuals (since the number of groups is significantly lower than the number of individuals)
