% Main Injector Design Code
% Gas Centred Swirl Injector Master Script %
% This Script Contains All Calculations for Deriving the Geometry of the Injector %
% For More Information See Design Document %
clc;
clear;
% N2O Jet Orifice %
N = 15; % number of injectors
mdot_N2O = 1.89740 / N; % mass flow rate of gaseous nitrous per injector (kg/s)
deltaP_N2O = 25e5; % required pressure drop across each N2O injector (Pa)
P = 25e5; % outlet pressure (Pa)
P0 = 50e5; % inlet pressure (Pa)
% Discharge Coefficient Estimate %
Re_guess = 1e6; % reynolds number estimate for N2O injection (can be quickly refined after running the script)
Cf = 0.0791 * Re_guess ^ -0.25; % fanning friction factor equation
Cd_N2O = 1 / sqrt((4 * Cf * 10) + 2.28); % estimation of N2O discharge coefficient (L/D = 10)
% List of Saturated Liquid N2O Properties %
% Entropy (J/mol.K) L
s_50 = 38.510;
s_45 = 36.759;
s_40 = 34.994;
s_35 = 33.021;
s_30 = 30.936;
s_25 = 28.617;
% Density (kg/m^3) L
rho_N2O_50 = 788.6;
rho_N2O_45 = 820.94;
rho_N2O_40 = 852.24;
rho_N2O_35 = 883.3;
rho_N2O_30 = 914.85;
rho_N2O_25 = 947.68;
% Ratio of Specific Heats L
gamma_50 = 3.3904;
gamma_45 = 3.0464;
gamma_40 = 2.7999;
gamma_35 = 2.6114;
gamma_30 = 2.4600;
gamma_25 = 2.3332;
% Enthalpy (J/kg) L
h_50 = 213.12e3;
h_25 = 147.51e3;
% List of Saturated Vapour N2O Properties %
% Entropy (J/mol.K) V
s_50v = 64.339;
s_45v = 65.589;
s_40v = 66.8;
s_35v = 68.013;
s_30v = 69.268;
s_25v = 70.614;
% Density (kg/m^3) V
rho_N2O_50v = 155.56;
rho_N2O_45v = 134.05;
rho_N2O_40v = 114.86;
rho_N2O_35v = 97.429;
rho_N2O_30v = 81.356;
rho_N2O_25v = 66.366;
% Ratio of Specific Heats V
gamma_50v = 3.0496;
gamma_45v = 2.6143;
gamma_40v = 2.3113;
gamma_35v = 2.0881;
gamma_30v = 1.9163;
gamma_25v = 1.77955;
% Enthalpy (J/kg) V
h_50v = 384.89e3;
h_25v = 400.12e3;
% Thermodynamic Path Calculations (Isentropic Flow) %
s_inlet = s_50; % entropy at the inlet, assuming saturated liquid at 50bar is inlet condition
s_outlet = s_inlet; % isentropic flow
X = (s_outlet - s_25) / (s_25v - s_25); % vapour fraction at the outlet
rho_outlet = (X * rho_N2O_25v) + ((1 - X) * rho_N2O_25); % density at the outlet
h_outlet = (X * h_25v) + ((1 - X) * h_25); % density at the outlet
% SPI Model %
G_SPI = Cd_N2O * sqrt(2 * rho_N2O_50 * deltaP_N2O); % mass flux
area_N2O_SPI = mdot_N2O / G_SPI; % orifice area
diameter_N2O_SPI = 2 * sqrt(area_N2O_SPI / pi); % orifice diameter
% SPC Model %
Y = sqrt(((P / P0)^(2/gamma_50)) * (gamma_50 / (gamma_50 - 1)) * ((1 - (P / P0)^((gamma_50 - 1) / gamma_50)) / (1 - (P / P0)))); % correction factor
G_SPC = G_SPI * Y; % mass flux
area_N2O_SPC = mdot_N2O / G_SPC; % orifice area
diameter_N2O_SPC = 2 * sqrt(area_N2O_SPC / pi); % orifice diameter
% HEM Model %
G_HEM = Cd_N2O * rho_outlet * sqrt(2*(h_50 - h_outlet)); % mass flux
area_N2O_HEM = mdot_N2O / G_HEM; % orifice area
diameter_N2O_HEM = 2 * sqrt(area_N2O_HEM / pi); % orifice diameter
% NHNE Model %
G_NHNE = (G_HEM + G_SPI) / 2; % mass flux
area_N2O_NHNE = mdot_N2O / G_NHNE; % orifice area
diameter_N2O_NHNE = 2 * sqrt(area_N2O_NHNE / pi); % orifice diameter
% FML Model %
slip = (rho_N2O_25 / rho_N2O_25v) ^ (1/3); % slip velocity
alpha = 1 / (1 + (((1 - X) / X) * slip * (rho_N2O_25v / rho_N2O_25)));
% void fraction
G_FML = ((1-alpha) * G_SPC) + (alpha * G_HEM); % mass flux
area_N2O_FML = mdot_N2O / G_FML; % orifice area
diameter_N2O_FML = 2 * sqrt(area_N2O_FML / pi); % orifice diameter
% FML is Model Used For Final Sizing %
% Diameter Console Output %
fprintf('FML Injector Diameter: %f mm\n', diameter_N2O_FML * 1e3);
% Additional Exit Parameters %
velocity_N2O = (mdot_N2O) / (rho_outlet * area_N2O_NHNE); % velocity (m/s)
fprintf('Outlet Velocity: %f m/s\n', velocity_N2O);
mu_25 = 102.33 * 1e-6;
mu_25v = 14.437 * 1e-6;
mu_outlet = (X * mu_25v) + ((1 - X) * mu_25); % dynamic viscosity (Pa.s)
fprintf('Outlet Dynamic Viscosity: %f μPa.s\n', mu_outlet * 1e6);
Re_outlet = (rho_outlet * velocity_N2O * diameter_N2O_FML) / (mu_outlet); % reynolds number at outlet
fprintf('Outlet Reynolds Number: %f million\n', Re_outlet * 1e-6);
% Mach Number Calculations %
pressureRatio = 45e5 / P0; % ratio of step
gamma = gamma_50;
pressureEquation = @(M_C) pressureRatio - (1 / ((1 + ((gamma - 1) / 2) * M_C^2) ^ (gamma / (gamma-1))));
M_initial = 0.2;
options = optimoptions('fsolve', 'Display', 'off');
[M_C, fval, exitflag] = fsolve(pressureEquation, M_initial, options);
fval45 = fval;
exitflag45 = exitflag;
M_C_45 = M_C;
T_45 = 292.69 / (1 + ((gamma - 1) / 2) * M_C^2);
gamma50 = gamma;
rho50 = rho_N2O_50;
pressureRatio = 40e5 / 45e5; % ratio of step
s_vap_45 = s_45v - s_45;
quality_outlet = (s_50 - s_45) / s_vap_45;
gamma = ((quality_outlet * gamma_45v) + ((1 - quality_outlet) * gamma_45));
pressureEquation = @(M_C) pressureRatio - ( ((1 + ((gamma - 1) / 2) * M_C_45^2)) / ((1 + ((gamma - 1) / 2) * M_C^2)) ) ^ (gamma / (gamma-1));
M_initial = 0.3;
options = optimoptions('fsolve', 'Display', 'off');
[M_C, fval, exitflag] = fsolve(pressureEquation, M_initial, options);
fval40 = fval;
exitflag40 = exitflag;
M_C_40 = M_C;
T_40 = 292.69 / (1 + ((gamma - 1) / 2) * M_C^2);
gamma45 = gamma;
rho45 = ((quality_outlet * rho_N2O_45v) + ((1 - quality_outlet) * rho_N2O_45));
pressureRatio = 35e5 / 40e5;
s_vap_40 = s_40v - s_40;
quality_outlet = (s_50 - s_40) / s_vap_40;
gamma = ((quality_outlet * gamma_40v) + ((1 - quality_outlet) * gamma_40));
pressureEquation = @(M_C) pressureRatio - ( ((1 + ((gamma - 1) / 2) * M_C_40^2)) / ((1 + ((gamma - 1) / 2) * M_C^2)) ) ^ (gamma / (gamma-1));
M_initial = 0.4;
options = optimoptions('fsolve', 'Display', 'off');
[M_C, fval, exitflag] = fsolve(pressureEquation, M_initial, options);
fval35 = fval;
exitflag35 = exitflag;
M_C_35 = M_C;
T_35 = 292.69 / (1 + ((gamma - 1) / 2) * M_C^2);
gamma40 = gamma;
rho40 = ((quality_outlet * rho_N2O_40v) + ((1 - quality_outlet) * rho_N2O_40));
pressureRatio = 30e5 / 35e5;
s_vap_35 = s_35v - s_35;
quality_outlet = (s_50 - s_35) / s_vap_35;
gamma = ((quality_outlet * gamma_35v) + ((1 - quality_outlet) * gamma_35));
pressureEquation = @(M_C) pressureRatio - ( ((1 + ((gamma - 1) / 2) * M_C_35^2)) / ((1 + ((gamma - 1) / 2) * M_C^2)) ) ^ (gamma / (gamma-1));
M_initial = 0.6; % Updated initial guess for M_C value
options = optimoptions('fsolve', 'Display', 'off');
[M_C, fval, exitflag] = fsolve(pressureEquation, M_initial, options);
fval30 = fval;
exitflag30 = exitflag;
M_C_30 = M_C;
T_30 = 292.69 / (1 + ((gamma - 1) / 2) * M_C^2);
gamma35 = gamma;
rho35 = ((quality_outlet * rho_N2O_35v) + ((1 - quality_outlet) * rho_N2O_35));
pressureRatio = 25e5 / 30e5;
s_vap_30 = s_30v - s_30;
quality_outlet = (s_50 - s_30) / s_vap_30;
quality_array(5) = quality_outlet;
gamma = ((quality_outlet * gamma_30v) + ((1 - quality_outlet) * gamma_30));
pressureEquation = @(M_C) pressureRatio - ( ((1 + ((gamma - 1) / 2) * M_C_30^2)) / ((1 + ((gamma - 1) / 2) * M_C^2)) ) ^ (gamma / (gamma-1));
M_initial = 0.7; % Updated initial guess for M_C value
options = optimoptions('fsolve', 'Display', 'off');
[M_C, fval, exitflag] = fsolve(pressureEquation, M_initial, options);
M_C_25 = M_C;
fval25 = fval;
exitflag25 = exitflag;
T_25 = 292.69 / (1 + ((gamma - 1) / 2) * M_C^2);
gamma30 = gamma;
rho30 = ((quality_outlet * rho_N2O_30v) + ((1 - quality_outlet) * rho_N2O_30));
fprintf('Outlet Mach Number: %f \n', M_C_25);
% IPA Swirl Orifice %
rho_IPA = 786; % density of IPA (kg/m^3)
mu_IPA = 0.0022; % dynamic viscosity of IPA (Pa.s)
mdot_IPA = 0.533646 / N; % mass flow rate of liquid IPA per injector (kg/s)
deltaP_IPA_sum = 5e5; % total pressure drop across each IPA injector (Pa)
% Manually Set Swirl Cone Half Angle %
% Vary This to Converge The r_I and r_II and phi_I and phi_II %
angle = 59.6304 * (pi / 180);
filling_eff_equation = @(filling_eff) ((2 * sqrt(2) * (1 - filling_eff)) / ((1 + sqrt(1 - filling_eff)) * sqrt(2 - filling_eff))) - sin(angle);
guess = 0.1;
options = optimoptions('fsolve', 'Display', 'off');
[filling_eff, fval, exitflag] = fsolve(filling_eff_equation, guess, options);
phi = filling_eff; % filling efficiency of injector
fprintf('phi_I: %f mm\n', phi);
diameter = diameter_N2O_FML; % inner orifice diameter
delta = 1e-3; % wall thickness 1mm
gap = 1e-3; % chosen gap thickness of 1mm
% For This Calculation We Will Fix a Nozzle Radius %
% Then Find the Vortex Chamber Radius, Inlet Radii, and Swirl Cone
Angle Which Can Converge The Calculations %
R = gap + ((diameter / 2) + delta); % fixed nozzle radius based on
desired gap
n = 3; % Number of tangential inlets
S = sqrt(1 - phi); % constant
% r_I Calculation %
LHS = (1 - phi) * sqrt(2) / (phi * sqrt(phi)); % define the left-hand
side of the equation (LHS)
f = @(r) LHS - (R - r) * R / (n * r^2); % define the function to find
the root of
r_guess = 1e-4; % define an initial guess for r
options = optimset('Display', 'off');
r_solution = fsolve(f, r_guess, options);
r = r_solution;
fprintf('r_I: %f mm\n', r * 1e3);
% r_II & phi_II Calculation %
X = 1 - phi; % passage emptiness
id = sqrt((X^3) / (2 - X));
cd = sqrt((phi ^ 3) / (2 - phi));
C = (2 * rho_IPA * deltaP_IPA_sum) / (mdot_IPA ^ 2);
Ao = pi * R ^ 2;
D = C - ((1 / (Ao * cd)) ^ 2);
Ai = sqrt(1 / (((n * id) ^ 2) * D));
r = sqrt((Ai / 1) / pi);
h = R * (1 - S); % film thickness
R_h = R - h;
phi_II = ((R ^ 2) - (R_h ^ 2)) / (R ^ 2);
fprintf('phi_II: %f mm\n', phi_II);
fprintf('r_II: %f mm\n', r * 1e3);
% Define Geometric Constant %
A = (sqrt(2) * (1 - phi)) / (phi * sqrt(phi));
% Now We Will Vary The Vortex Chamber Radius So The Flow Equations Used in r_II Calculations are Valid %
Pdrop_T = ((mdot_IPA / n) / (Ai * id * sqrt(2 * rho_IPA))) ^ 2; % tangential inlet pressure drop for ideal flow
Pdrop_V = (mdot_IPA / (Ao * cd * sqrt(2 * rho_IPA))) ^ 2; % vortex chamber pressure drop for ideal flow
fprintf('Pv: %f bar\n', Pdrop_V * 1e-5);
fprintf('Pt: %f bar\n', Pdrop_T * 1e-5);
R_V = 1.1171524 * R; % vortex chamber radius (manually varied)
Ratio = ((2 * (1 - phi) ^ 2) / (2 - phi)) / (((R_V - r) / R) ^ 2); % equation for this ratio => (deltaP_T / delta_P_sum)
P_T = deltaP_IPA_sum * Ratio; % real tangential inlet pressure drop
% We Vary R_V So the Real Value of P_T is Equal to the Ideal Value %
fprintf('Pt: %f bar\n', P_T * 1e-5);
Pdrop_calc = Pdrop_V + Pdrop_T; % total pressure drop check, for converged geometry this should be equal to deltaP_IPA_sum
fprintf('Total P Drop: %f bar\n', Pdrop_calc * 1e-5);
% Console Outputs %
fprintf('Swirl Cone Angle: %f degrees\n', 2 * angle * (180 / pi));
fprintf('Film Thickness Nozzle: %f mm\n', h * 1e3);
fprintf('Nozzle Injector Diameter: %f mm\n', 2 * R * 1e3);
fprintf('Injector Vortex Chamber Diameter: %f mm\n', 2 * R_V * 1e3);
fprintf('Inlet Diameters: %f mm\n', 2 * r * 1e3);
fprintf('Filling Efficiency: %f \n', phi);
fprintf('Nozzle Flow Coefficient: %f \n', cd);
h_V = (R_V - R * sqrt( ( ((A^2) * (cd^2)) / (1 - (phi / (3 - 2 * phi))))));
fprintf('Film Thickness Vortex Chamber: %f mm\n', h_V * 1e3);
h_ex = R * (1 - sqrt(1 - (phi / sqrt(3 - (2 * phi)))));
fprintf('Film Thickness Exit: %f mm\n', h_ex * 1e3);
h_V0 = (R_V - (R * A * cd));
fprintf('Film Thickness Initial: %f mm\n', h_V0 * 1e3);
u_sum = sqrt((2 * deltaP_IPA_sum) / rho_IPA);
u_axV = (phi / sqrt(3 - 2 * phi)) * u_sum;
fprintf('Axial Velocity Vortex Chamber: %f m/s\n', u_axV);
u_axN = sqrt(phi / (2 - phi)) * u_sum;
fprintf('Axial Velocity Nozzle: %f m/s\n', u_axN);
u_axE = sqrt(3 - 2 * phi) * u_axN;
fprintf('Axial Velocity Exit: %f m/s\n', u_axE);
Q = phi * Ao * u_axN;
v_tangin = Q / (n * pi * r ^ 2);
fprintf('Tangential Inlet Velocity: %f m/s\n', v_tangin);
MR = (rho_IPA * u_axE ^ 2) / (rho_outlet * velocity_N2O ^ 2);
MR = 1 / MR;
fprintf('Momentum Flux Ratio: %f \n', MR);
% Length Calculations %
RR = 1.5; % recess ratio
recess = RR * diameter_N2O_FML; % recess length
LN = 1 * R; % nozzle length (in reality LN will be set to recess +0.1mm)
LV_min = 2 * (R_V - r); % minimum vortex chamber length (in reality this is set to ~6mm)
LT = 3.5 * r; % rough tangential inlet length
LN2O = 10 * diameter_N2O_FML; % jet length
fprintf('recess: %f mm\n', recess * 1e3);
fprintf('LN: %f mm\n', LN * 1e3);
fprintf('LV_min: %f mm\n', LV_min * 1e3);
fprintf('LT_rough: %f mm\n', LT * 1e3);
fprintf('LN2O: %f mm\n', LN2O * 1e3);
