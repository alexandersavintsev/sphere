import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a, b = 0.40, 0.45             
Ta, Tb = 100.0, 0.0             
E, nu = 200e9, 0.30              
alpha = 1.2e-5                    

mu = E/(2*(1+nu))                 
K  = E/(3*(1-2*nu))                

C1_T = (b*Tb - a*Ta)/(b - a)
C2_T = (a*b*(Ta - Tb))/(b - a)

def T(r):
    return C1_T + C2_T/r

def I(r):
    return (C1_T/3.0)*(r**3 - a**3) + (C2_T/2.0)*(r**2 - a**2)

beta = (3*K*alpha) / (K + 4.0*mu/3.0)

Ib = I(b)
C1 = (beta*Ib) / ((3*K/(4*mu))*b**3 + a**3)
C2 = -C1*a**3

def u(r):
    return C1*r + C2/r**2 + (beta/r**2)*I(r)

def tau_rr(r):
    return 3*K*C1 - (4*mu/r**3)*(C2 + beta*I(r))

r = np.linspace(a, b, 400)
u_vals = u(r)
trr_vals = tau_rr(r)

df = pd.DataFrame({
    "r_m": r,
    "u_m": u_vals,
    "tau_rr_Pa": trr_vals,
    "T_C": T(r)
})
df.to_csv("thermal_sphere_results.csv", index=False)

plt.figure(figsize=(6.2,4.1))
plt.plot(r, u_vals*1e6, lw=2)
plt.xlabel("r, м"); plt.ylabel("u, мкм"); plt.title("Перемещения u(r)")
plt.grid(True); plt.tight_layout(); plt.savefig("u_of_r.png", dpi=200)

plt.figure(figsize=(6.2,4.1))
plt.plot(r, trr_vals/1e6, lw=2)
plt.xlabel("r, м"); plt.ylabel(r"$\tau_{rr}$, МПа"); plt.title("Эпюра радиальных напряжений $\\tau_{rr}(r)$")
plt.grid(True); plt.tight_layout(); plt.savefig("tau_rr.png", dpi=200)

# ---- Print control values to console ----
print("----- CONTROL NUMBERS -----")
print(f"K    = {K/1e9:8.3f} GPa,  mu = {mu/1e9:8.3f} GPa,  beta = {beta:.3e}")
print(f"I(b) = {Ib:.6f}  (m^3*°C)")
print(f"C1   = {C1:.6e},  C2 = {C2:.6e}")
print(f"u(b)           = {u_vals[-1]*1e6:7.2f}  µm")
print(f"tau_rr(a)      = {tau_rr(a+1e-6)/1e6:7.2f}  MPa")
print(f"tau_rr(b)      = {tau_rr(b)/1e6:7.2f}  MPa")
print("---------------------------")
print("Files written: u_of_r.png, tau_rr.png, thermal_sphere_results.csv")
