# %%
import numpy as np
import matplotlib.pyplot as plt

class Strategy:
    #Klasa bazowa (interfejs) wymuszająca implementację metody 'choose_price' przez wszystkie podklasy.
    def __init__(self):
        pass
    def choose_price(self, market, agent):
        raise NotImplementedError("Every strategy must implement 'choose_price' method.")

class Aggressive(Strategy):
    def choose_price(self, market, agent):
        if market.r == agent.initial_round: 
            return max(market.mc, market.price_mono() - market.eps)
        else:
            return max(market.mc, market.prev_p_min - market.eps)

class Imitation(Strategy):
    def choose_price(self, market, agent):
        if market.r == agent.initial_round: 
            return market.price_mono()
        else:
            return market.prev_p_min

class Cooperation(Strategy):
    def choose_price(self, market, agent):
        return market.price_mono()

class ConditionalCooperation(Strategy):
    def choose_price(self, market, agent):
        if market.r == agent.initial_round: 
            return market.price_mono()
        elif market.prev_p_min_star >= market.price_mono():
            return market.price_mono()
        else: 
            return max(market.mc, market.prev_p_min_star)

class Stochastic(Strategy):
    def choose_price(self, market, agent):
        if market.r == agent.initial_round: 
            return np.random.uniform(market.mc, market.price_mono())
        else:
            alpha = 3.0 
            p_sup = market.price_mono() + market.eps
            x_mode = (agent.price - market.mc)/(p_sup - market.mc)
            # Zabezpieczenie przed wyjściem mody poza przedział (0, 1), chroni przed błędem parametrów rozkładu Beta
            x_mode = np.clip(x_mode, 0.001, 0.999)
            # Wyznaczenie parametru beta, wymuszające położenie maksimum gęstości w punkcie x_mode
            beta = (alpha - 1)/x_mode - alpha + 2
            x_beta =  np.random.beta(alpha, beta)
            return (p_sup - market.mc)*x_beta + market.mc

class Adaptive(Strategy):
    def choose_price(self, market, agent):
         if market.r == agent.initial_round: 
            return market.price_mono()
         else:
             return market.prev_market_leader_p

class GrimTrigger(Strategy):
    def choose_price(self, market, agent):
        if market.prev_p_min < market.price_mono():
            agent.coop_is_broken = True
        if market.r == agent.initial_round or not agent.coop_is_broken: 
            return market.price_mono()
        return market.mc

class WinStayLoseShift(Strategy):
    def choose_price(self, market, agent):
        if market.r == agent.initial_round: 
            return np.random.uniform(market.mc, market.price_mono())
        elif agent.prev_profit >= market.fc:
            return agent.price
        else:
            return np.random.uniform(market.mc, market.price_mono() + market.eps)

class Agent:
    #Reprezentuje jednostkę rynkową przechowującą stan konta, zysk z poprzedniej rundy oraz instancję strategii.
    def __init__(self, saldo: float, initial_round: int, strategy: Strategy):
        self.price = market.mc
        self.strategy = strategy
        self.saldo = saldo
        self.prev_profit = 0
        self.initial_round = initial_round
        self.coop_is_broken = False

    def choose_price(self, market):
        return self.strategy.choose_price(market, self)

class Market:
    #Silnik symulacji zarządzający populacją agentów, przepływem iteracji i kalkulacją parametrów globalnych.
    def __init__(self, fc: float, K: float, mc:float, mu:float, m:int, A:float, a:float, eps: float, agents:list[Agent]):
        self.fc = fc
        self.mc = mc
        self.eps = eps
        self.agents = agents
        self.mu = mu
        self.m = m
        self.A = A
        self.a = a
        self.r = 1
        self.K = K
        self.prev_p_min = self.price_mono()
        self.prev_p_min_star = self.price_mono()
        self.prev_market_leader_p = self.price_mono()

    def round(self):
        """
        Wykonuje pełną sekwencję zdarzeń w iteracji:
        1. Ewolucja populacji (tylko w rundach będących wielokrotnością parametru m).
        2. Rozliczenie finansowe: pobranie zysków, potrącenie kosztów stałych.
        3. Identyfikacja cen historycznych liderów (wymagane dla strategii adaptacyjnej).
        4. Wybór nowych cen przez wszystkich agentów.
        5. Alokacja wolumenu i wyliczenie zysków dla agentów dyktujących cenę minimalną.
        """
        if self.r % self.m == 0:
            self.evolve()

        for agent in self.agents:
            agent.saldo = agent.saldo + agent.prev_profit - self.fc

        max_balance = max(agent.saldo for agent in self.agents)
        M_bal = []
        for agent in self.agents:
            if agent.saldo == max_balance: 
                M_bal.append(agent)
                self.prev_market_leader_p = agent.price
        for agent in M_bal:
            if agent.price > self.prev_market_leader_p: 
                self.prev_market_leader_p = agent.price

        for agent in self.agents:
            agent.price = agent.choose_price(self)

        M_k = [] 
        p_min = min(agent.price for agent in self.agents)
        self.prev_p_min = p_min
        market_volume = self.market_volume(p_min)
        for agent in self.agents:
            if agent.price == p_min:
                M_k.append(agent)
        for agent in self.agents:
            if agent in M_k: 
                agent.prev_profit = (agent.price - self.mc)*(market_volume/len(M_k))
            else: agent.prev_profit = 0
        self.r += 1

    def evolve(self):
        """
        Mechanizm selekcji naturalnej. 
        Wycofuje z rynku bankrutów (saldo < 0) i zastępuje ich podmiotami z odnowionym kapitałem.
        Strategie nowych agentów są kopiowane od liderów lub losowane ze zbioru (z prawdopodobieństwem mu).
        """
        winners = []
        losers = []
        for agent in self.agents:
            if agent.saldo >= 0:
                winners.append(agent)
            else:
                losers.append(agent)
        new_pop = []
        S = [Aggressive, Imitation, Cooperation, ConditionalCooperation, Stochastic, Adaptive, GrimTrigger, WinStayLoseShift]
        if winners:
            self.prev_p_min_star = min(agent.price for agent in winners)
            max_balance = max(agent.saldo for agent in winners)
            liders = []
            for agent in winners:
                if agent.saldo == max_balance: 
                    liders.append(agent)    
            S1 = [type(agent.strategy) for agent in liders]
            S2 = [cls for cls in S if cls not in S1]
            for _ in range(len(losers)):
                mu1 = np.random.rand()
                if mu1 >= self.mu:
                    st = np.random.choice(S1)
                else:
                    st = np.random.choice(S2)
                new_pop.append(Agent(self.K, self.r, st()))
        else:
            self.prev_p_min_star = self.price_mono()
            for _ in range(len(losers)):
                st = np.random.choice(S)
                new_pop.append(Agent(self.K, self.r, st()))
        self.agents = new_pop + winners

    def market_volume(self, p: float):
        #Oblicza dostępny wolumen rynkowy ze ściśle malejącej wykładniczej funkcji popytu.
        return self.A*np.exp(-self.a*p)

    def price_mono(self):
        #Analitycznie wyznacza cenę monopolową maksymalizującą zagregowany zysk operacyjny układu.
        return self.mc + 1/self.a
    
if __name__ == "__main__":
    market = Market(20, 500, 5, 0.5, 5, 100, 0.1, 0.01, [])
    
    starting_agents = [
        Agent(500, 1, Aggressive()),
        Agent(500, 1, Imitation()),
        Agent(500, 1, Cooperation()),
        Agent(500, 1, GrimTrigger()),
        Agent(500, 1, ConditionalCooperation()),
        Agent(500, 1, Stochastic()),
        Agent(500, 1, Adaptive()),
        Agent(500, 1, WinStayLoseShift())
    ]
    market.agents = starting_agents

    rounds_hist = []
    prices_hist = []
    avg_balances_hist = []
    max_balances_hist = []

    strategy_names = ["Aggressive", "Imitation", "Cooperation", "ConditionalCooperation", "Stochastic", "Adaptive", "GrimTrigger", "WinStayLoseShift"]
    strategy_counts_hist = {name: [] for name in strategy_names}

    for _ in range(10000):
        rounds_hist.append(market.r)
        
        market.round()
        
        prices_hist.append(market.prev_p_min)
        
        current_balances = [agent.saldo for agent in market.agents]
        avg_balances_hist.append(np.mean(current_balances))
        max_balances_hist.append(np.max(current_balances))
        
        current_counts = {name: 0 for name in strategy_names}
        for agent in market.agents:
            current_counts[type(agent.strategy).__name__] += 1
            
        for name in strategy_names:
            strategy_counts_hist[name].append(current_counts[name])

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(rounds_hist, prices_hist, label="Cena minimalna rynkowa", color="black", linewidth=2)
    ax1.axhline(market.price_mono(), color="red", linestyle="--", label="Cena monopolowa")
    ax1.axhline(market.mc, color="blue", linestyle=":", label="Koszt krańcowy (MC)")
    ax1.set_xlabel("Runda")
    ax1.set_ylabel("Cena")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(rounds_hist, avg_balances_hist, label="Średnie saldo w populacji", color="green", linewidth=2)
    ax2.plot(rounds_hist, max_balances_hist, label="Maksymalne saldo lidera", color="blue", linestyle="--")
    ax2.set_xlabel("Runda")
    ax2.set_ylabel("Saldo")
    ax2.legend(loc='upper left')
    ax2.grid(True)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    counts_matrix = np.array([strategy_counts_hist[name] for name in strategy_names])
    ax3.stackplot(rounds_hist, counts_matrix, labels=strategy_names, alpha=0.8)
    ax3.set_xlabel("Runda")
    ax3.set_ylabel("Liczba agentów")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True)
    fig3.tight_layout()

    plt.show() 
# %%
