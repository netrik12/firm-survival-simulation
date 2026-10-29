# Agent-Based Simulation of Firm Survival in Price-Elastic Markets

An object-oriented simulation in Python analyzing evolutionary stability and pricing dynamics in a Bertrand oligopoly with exponential demand.

## Overview
This project simulates competitive market dynamics where firms repeatedly choose pricing strategies under operational fixed costs ($FC$) and marginal costs ($MC$). Under classical assumptions, price competition collapses into the Bertrand Paradox ($P = MC$), leading to systemic bankruptcy. 

This model introduces an evolutionary selection framework with 8 heterogeneous agent strategies to observe market survival, price wars, and the endogenous emergence of **tacit collusion**.

## Implemented Strategies
- **Aggressive**: Undercuts market minimum by $\epsilon$.
- **Imitation**: Copies the previous minimum price.
- **Cooperation / Conditional Cooperation**: Targets monopoly price; retaliates if undercut.
- **Grim Trigger**: Punishes deviations by permanently setting price to $MC$.
- **Stochastic (Beta Distribution)**: Dynamically calibrated Beta density centered on the agent's prior price.
- **Adaptive**: Copies the price of the most capitalized firm.
- **Win-Stay, Lose-Shift (WSLS)**: Maintains price on positive profit, randomizes on loss.

## Key Findings
- Aggressive pricing strategies consistently drive the market into systemic insolvency.
- Elimination of destructive strategies allows reactive agents (Imitation, Adaptive) to endogenously coordinate on the monopoly price without explicit communication.
- Evolutionary pressure stabilizes long-term capital accumulation once price wars subside.

## Tech Stack
- **Python 3.10+**
- **NumPy** (numerical operations, stochastic sampling)
- **Matplotlib** (market dynamic visualizations)

## How to Run
```bash
pip install numpy matplotlib
python model_ec_APP.py
