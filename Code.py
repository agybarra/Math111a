import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# 1. Define parameters for each T_i(t)
# -----------------------------
params = {
    "food":      {"k": 0.220, "t0": 1970, "w": 0.166},
    "water":     {"k": 0.163, "t0": 1908, "w": 0.166},
    "shelter":   {"k": 0.219, "t0": 1989, "w": 0.166},
    "wage":      {"k": 0.071, "t0": 1970, "w": 0.166},
    "education": {"k": 0.044, "t0": 1950, "w": 0.166},
    "healthcare":{"k": 0.055, "t0": 1960, "w": 0.166},
}

C = 0.103      # scaling factor for carrying capacity
k_min = 1      # minimum carrying capacity in billions

# -----------------------------
# Logistic technology function T_i(t)
# -----------------------------
def T_i(t, k, t0):
    return 100 / (1 + np.exp(-k * (t - t0)))

# -----------------------------
# Welfare function W(t)
# -----------------------------
def W(t):
    return sum(p["w"] * T_i(t, p["k"], p["t0"]) for p in params.values())

# -----------------------------
# Dynamic carrying capacity k(t)
# -----------------------------
def K(t):
    return k_min + C * W(t)

# -----------------------------------
# Intrinsic growth rate α(t)
# -----------------------------------

alpha_base = 0.004    # growth at start year
alpha_peak = 0.018    # demographic boom peak
mu = 1963             # peak year
sigma = 25            # width

def alpha(t):
    return alpha_base + alpha_peak * np.exp(-(t - mu)**2 / (2 * sigma**2))

# -----------------------------
# Population ODE
# -----------------------------
P0 = 1    # population in 1800 (billions)
t_span = (1800, 2100)
t_eval = np.linspace(1800, 2100, 3000)

def dP_dt(t, P):
    return alpha(t) * P * (1 - P / K(t))   # <-- FIXED: uses alpha(t)

sol = solve_ivp(dP_dt, t_span, [P0], t_eval=t_eval)
P_vals = sol.y[0]

# -----------------------------
# Plot T_i(t)
# -----------------------------
plt.figure(figsize=(12,7))
for name, p in params.items():
    plt.plot(t_eval, T_i(t_eval, p["k"], p["t0"]), linewidth=2, label=name)

plt.title("Technology Functions $T_i(t)$", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Percent Fulfillment (%)", fontsize=14)
plt.xlim(1800, 2100)
plt.ylim(-5, 105)
plt.grid(alpha=0.3, linestyle="--")
plt.legend()
plt.savefig("T_i.png")


# -----------------------------
# Plot W(t)
# -----------------------------
W_vals = [W(t) for t in t_eval]

plt.figure(figsize=(12,7))
plt.plot(t_eval, W_vals, linewidth=3, color="black")
plt.title("Welfare-Based Index $W(t)$", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Welfare Index (0–100)", fontsize=14)
plt.xlim(1800, 2100)
plt.ylim(0, 100)
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("W(t).png")

# -----------------------------
# Plot K(t)
# -----------------------------
K_vals = [K(t) for t in t_eval]

plt.figure(figsize=(12,7))
plt.plot(t_eval, K_vals, linewidth=3, color="darkorange")
plt.title("Dynamic Carrying Capacity $K(t)$", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Carrying Capacity (billions)", fontsize=14)
plt.xlim(1800, 2100)
plt.ylim(min(K_vals)-0.5, max(K_vals)+0.5)
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("K(t)).png")

# -----------------------------
# Plot α(t)
# -----------------------------
plt.figure(figsize=(12,7))
plt.plot(t_eval, alpha(t_eval), linewidth=3, color="blue")
plt.title("Intrinsic Growth Rate α(t)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Growth Rate", fontsize=14)
plt.xlim(1800, 2100)
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("α(t).png")

# -----------------------------
# Plot Population P(t)
# -----------------------------
plt.figure(figsize=(12,7))
plt.plot(t_eval, P_vals, linewidth=3, color="green")
plt.title("Population Curve with Dynamic α(t) + Welfare Capacity K(t)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Population (billions)", fontsize=14)
plt.xlim(1800, 2100)
plt.ylim(0, max(P_vals)+1)
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("P(t).png")

# -----------------------------
# Contribution Analysis
# -----------------------------
plt.stackplot(
    t_eval,
    [p["w"] * T_i(t_eval, p["k"], p["t0"]) for p in params.values()],
    labels=params.keys()
)
plt.legend()
plt.title("Contribution of Technologies to Welfare W(t)")
plt.savefig("Contributions.png")


# -----------------------------
# Population Relative to Carrying Capacity
# -----------------------------
ratio_vals = P_vals / np.array(K_vals)

plt.figure(figsize=(12,7))
plt.plot(t_eval, ratio_vals, color="red", linewidth=2)
plt.title("Population Utilization of Carrying Capacity P(t)/K(t)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Utilization Ratio", fontsize=14)
plt.ylim(0, 1.5)
plt.grid(alpha=0.3, linestyle="--")
plt.axhline(1, color="black", linestyle="--", alpha=0.7)  # threshold line
plt.savefig("Utilization.png")


# -----------------------------
# Scenario Comparisons
# -----------------------------
# Case A: Constant α, Static K
alpha_const = 0.01
K_const = 10
def dP_dt_const(t, P): return alpha_const * P * (1 - P / K_const)
sol_const = solve_ivp(dP_dt_const, t_span, [P0], t_eval=t_eval)

# Case B: Dynamic α(t), Static K
def dP_dt_dyn_alpha(t, P): return alpha(t) * P * (1 - P / K_const)
sol_dyn_alpha = solve_ivp(dP_dt_dyn_alpha, t_span, [P0], t_eval=t_eval)

# Case C: Constant α, Dynamic K(t)
def dP_dt_dyn_K(t, P): return alpha_const * P * (1 - P / K(t))
sol_dyn_K = solve_ivp(dP_dt_dyn_K, t_span, [P0], t_eval=t_eval)

# Case D: Dynamic α(t), Dynamic K(t) (your main model)
sol_main = sol

plt.figure(figsize=(12,7))
plt.plot(t_eval, sol_const.y[0], label="Const α, Const K", linestyle=":")
plt.plot(t_eval, sol_dyn_alpha.y[0], label="Dyn α, Const K", linestyle="--")
plt.plot(t_eval, sol_dyn_K.y[0], label="Const α, Dyn K", linestyle="-.")
plt.plot(t_eval, sol_main.y[0], label="Dyn α, Dyn K", linewidth=3, color="green")

plt.title("Scenario Comparisons of Population Growth", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Population (billions)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("Comparisons.png")


# -----------------------------
# Sensitivity Analysis
# -----------------------------
C_values = [0.05, 0.103, 0.2]          # different carrying capacity scaling
alpha_peaks = [0.012, 0.018, 0.025]    # different demographic booms

plt.figure(figsize=(12,7))

for C_test in C_values:
    def K_test(t): return k_min + C_test * W(t)
    def dP_dt_test(t, P): return alpha(t) * P * (1 - P / K_test(t))
    sol_test = solve_ivp(dP_dt_test, t_span, [P0], t_eval=t_eval)
    plt.plot(t_eval, sol_test.y[0], label=f"C={C_test}")

for a_peak in alpha_peaks:
    def alpha_test(t): return alpha_base + a_peak * np.exp(-(t - mu)**2 / (2 * sigma**2))
    def dP_dt_test(t, P): return alpha_test(t) * P * (1 - P / K(t))
    sol_test = solve_ivp(dP_dt_test, t_span, [P0], t_eval=t_eval)
    plt.plot(t_eval, sol_test.y[0], linestyle="--", label=f"α_peak={a_peak}")

plt.title("Sensitivity Analysis of Population Curves", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Population (billions)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("SensitivityAnalysis.png")


# -----------------------------
# Welfare Index Comparison: Equal vs Custom Weights
# -----------------------------
# Equal weights (already defined)
W_equal = [W(t) for t in t_eval]

custom_weights = {
    "food": 0.30,
    "water": 0.25,
    "shelter": 0.15,
    "wage": 0.10,
    "education": 0.10,
    "healthcare": 0.10,
}

def W_custom(t):
    return sum(custom_weights[name] * T_i(t, p["k"], p["t0"]) for name, p in params.items())

W_custom_vals = [W_custom(t) for t in t_eval]

plt.figure(figsize=(12,7))
plt.plot(t_eval, W_equal, label="Equal Weights", linewidth=3, color="black")
plt.plot(t_eval, W_custom_vals, label="Custom Weights", linewidth=3, color="red", linestyle="--")
plt.title("Welfare Index Comparison: Equal vs Custom Weights", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Welfare Index (0–100)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("W(t)CustomvsEqual.png")


# -----------------------------
# Compare P(t) to Historical + Future Projections (Line + Simulated Curve)
# -----------------------------
# Historical + UN projection data (billions, approximate UN World Population Prospects)
years_data = [1800, 1850, 1900, 1950, 2000, 2020, 2050, 2100]
pop_data   = [1.0, 1.2, 1.6, 2.5, 6.1, 7.8, 9.7, 10.4]  # billions

plt.figure(figsize=(12,7))
plt.plot(t_eval, P_vals, label="Simulated Population", linewidth=3, color="green")
plt.plot(years_data, pop_data, label="Historical + UN Projection", linewidth=2, color="blue", marker="o")
plt.title("Simulated Population vs Historical and Projected Data", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Population (billions)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("P(t)toHistorical.png")


# -----------------------------
# Population vs Welfare Thresholds (80 and 90)
# -----------------------------
above_70 = np.array(W_vals) > 70
above_80 = np.array(W_vals) > 80
above_90 = np.array(W_vals) > 90

t70 = t_eval[np.where(above_70)[0][0]]
t80 = t_eval[np.where(above_80)[0][0]]
t90 = t_eval[np.where(above_90)[0][0]]

plt.figure(figsize=(12,7))
plt.plot(t_eval, P_vals, color="green", linewidth=2, label="Population P(t)")
plt.fill_between(t_eval, P_vals, 0, where=above_70, color="pink", alpha=0.3, label="W(t) > 70")
plt.fill_between(t_eval, P_vals, 0, where=above_80, color="orange", alpha=0.3, label="W(t) > 80")
plt.fill_between(t_eval, P_vals, 0, where=above_90, color="blue", alpha=0.3, label="W(t) > 90")


plt.title("Population Periods Above Welfare Thresholds", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Population (billions)", fontsize=14)
plt.grid(alpha=0.3, linestyle="--")
plt.axvline(t70, color="pink", linestyle="--", label=f"W>70 at year {int(t70)}")
plt.axvline(t80, color="orange", linestyle="--", label=f"W>80 at year {int(t80)}")
plt.axvline(t90, color="blue", linestyle="--", label=f"W>90 at year {int(t90)}")
plt.legend()
plt.savefig("popvsThresh.png")

# -----------------------------
# Plot Population P(t) and Carrying Capacity K(t) on Same Graph
# -----------------------------
plt.figure(figsize=(12,7))
plt.plot(t_eval, P_vals, linewidth=3, color="green", label="Population P(t)")
plt.plot(t_eval, K_vals, linewidth=3, color="darkorange", linestyle="--", label="Carrying Capacity K(t)")

plt.title("Population vs Dynamic Carrying Capacity", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Billions", fontsize=14)
plt.xlim(1800, 2100)
plt.ylim(0, max(max(P_vals), max(K_vals)) + 1)
plt.legend()
plt.grid(alpha=0.3, linestyle="--")
plt.savefig("popvsCC.png")


