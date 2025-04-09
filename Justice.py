import streamlit as st
import nashpy as nash
import numpy as np
from scipy.optimize import linprog
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import random

# -------------------------- Streamlit App --------------------------
st.set_page_config(page_title="SDG 16 Optimizer", layout="wide")
st.title("SDG 16 Optimizer: Peace, Justice & Strong Institutions")

st.sidebar.header("Optimization Controls")

# -------------------------- Nash Game Theory --------------------------
st.subheader("1. Conflict Resolution - Game Theory")

A_payoff = np.array([[3, 1], [0, 2]])
B_payoff = np.array([[2, 0], [1, 3]])
game = nash.Game(A_payoff, B_payoff)
equilibria = list(game.support_enumeration())

st.markdown("*Nash Equilibria:*")
for eq in equilibria:
    st.write(f"Group A Strategy: {np.round(eq[0],2)} | Group B Strategy: {np.round(eq[1],2)}")

# -------------------------- Linear Programming --------------------------
st.subheader("2. Resource Allocation - Linear Programming")

c = [-10, -15, -20]
A = [[1, 1, 1], [2, 1, 3], [1, 3, 2]]
b = [100, 180, 150]
x_bounds = [(0, 50), (0, 40), (0, 30)]
res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, method='highs')

if res.success:
    st.markdown("*Optimal Allocation (Institution A, B, C):*")
    st.write(np.round(res.x, 2))
    st.markdown(f"*Total Impact Score:* {round(-res.fun, 2)}")
else:
    st.error("Linear programming failed.")

# -------------------------- Genetic Algorithm --------------------------
st.subheader("3. Strategy Optimization - Genetic Algorithm")

gen_count = st.sidebar.slider("GA Generations", 10, 100, 40)
pop_size = st.sidebar.slider("GA Population Size", 10, 100, 30)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: random.uniform(0, 1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalInstitution(ind):
    score = 10*ind[0] + 15*ind[1] + 20*ind[2]
    budget = ind[0] + ind[1] + ind[2]
    manpower = 2*ind[0] + 1*ind[1] + 3*ind[2]
    training = 1*ind[0] + 3*ind[1] + 2*ind[2]
    penalty = 0
    if budget > 1 or manpower > 1.8 or training > 1.5:
        penalty = 50
    return score - penalty,

toolbox.register("evaluate", evalInstitution)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=pop_size)
algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=gen_count, verbose=False)
top = tools.selBest(pop, k=1)[0]

st.markdown("*Best Institutional Weights (A, B, C):*")
st.write([round(x, 2) for x in top])
st.markdown(f"*GA Impact Score:* {round(evalInstitution(top)[0], 2)}")

# -------------------------- Pareto Analysis --------------------------
st.subheader("4. Multi-Objective Visualization - Pareto Front")

X = np.linspace(0, 1, 50)
Y = 1 - X
Z = [10*x + 15*y for x, y in zip(X, Y)]

fig, ax = plt.subplots()
ax.plot(X, Z, color='green', label="Peace vs Institutions Tradeoff")
ax.set_xlabel("Conflict Mitigation Effort (Normalized)")
ax.set_ylabel("Institutional Impact Score")
ax.set_title("Pareto Analysis for SDG 16")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# -------------------------- Footer --------------------------
st.markdown("---")
st.markdown("Developed as a prototype for *SDG 16* strategies using optimization and AI. Contact for collaboration or full-scale simulation.")