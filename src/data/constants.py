# ============================================================
# Concrete TES Properties
# ============================================================

# Density of concrete (rho_con) [kg/m^3]
rho_con = 2340

# Specific heat of concrete (Cp_con) [J/(kg·K)]
Cp_con = 920

# Thermal conductivity of concrete (lambda_con) [W/(m·K)]
lambda_con = 1.40

# Internal pipe diameter (D_int) [m]
D_int = 0.021

# External pipe diameter (D_ext) [m]
D_ext = 0.025

# Thermal conductivity of insulation (lambda_iso) [W/(m·K)]
lambda_iso = 0.06

# Thickness of the insulation (th_iso) [m]
th_iso = 0.30

# Emissivity of insulation (epsilon_iso) [-]
epsilon_iso = 0.05

# ============================================================
# CTES Geometry
# ============================================================

# Volume of concrete in the CTES [m^3]
V_con = 1319

# Number of CTES modules in series
n_modules = 14

# CTES module dimensions (including insulation) [m]
module_width = 2
module_height = 4
module_length = 18

# LFC mirror surface [m^2]
A_LFC = 6602

# Number of LFC modules
n_LFC_modules = 300

# Number of LFC rows
n_LFC_rows = 20

# Area per LFC module [m^2]
A_LFC_module = 30.50

# Total soil surface [m^2]
A_soil_total = 10100

# ------------------------------------------------------------------
# Fluid and operating properties (moved from ctes_1d_jian)
# ------------------------------------------------------------------
# Paratherm-NF (typical reference properties around 200 C)
rho_f = 780.0    # [kg/m3]
Cp_f  = 2200.0   # [J/(kg·K)]
mu_f  = 0.5e-3   # [Pa·s]
k_f   = 0.12     # [W/(m·K)]

# Operating temperatures and environment
T_min = 130.0    # [C]  minimum operating temperature (return oil)
T_max = 310.0    # [C]  maximum operating temperature (from solar field)
T_amb = 30.0     # [C]  ambient temperature
v_wind = 3.0     # [m/s]
sigma = 5.670e-8 # [W/(m2·K4)]

# Total oil flow rate (Buscemi et al.)
Q_vol_total = 0.019                       # [m3/s] total volumetric flow rate

# Equivalent element count used for scaling
n_pipes = 818
