clc; clear;

id_1_4_tube = in_2_m(0.25 - 2*0.036); % m
id_3_8_tube = in_2_m(0.375 - 2*0.036); % m % m
id_1_2_tube = in_2_m(0.5 - 2*0.036); % m
id_3_4_tube = in_2_m(0.75 - 2*0.036); % m

A_1_4_tube = id_2_A(id_1_4_tube); % m^2
A_3_8_tube = id_2_A(id_3_8_tube); % m^2
A_1_2_tube = id_2_A(id_1_2_tube); % m^2
A_3_4_tube = id_2_A(id_3_4_tube); % m^2

V_pressurant_tank = 6.8e-3; % m^3
% V_pressurant_tank = 9e-3; % m^3
V_fuel_tank = 7.4e-3; % m^3
V_ox_tank = 18.8e-3; % m^3

A_tank = id_2_A(180e-3); % m^2

pressurant_init_T = 293.15; % K
pressurant_init_P = 250e5; % Pa

fuel_init_T = 293.15; % K
fuel_init_P = 50e5; % Pa

ox_init_T = 293.15; % K
ox_init_P = 50e5; % Pa

rho_fuel = 790; % kg/m^3
rho_ox = 950; % kg/m^3

ullage_frac_init = 0.1; % fraction

V_fuel_ullage_init = ullage_frac_init * V_fuel_tank; % m^3
V_fuel_init = V_fuel_tank - V_fuel_ullage_init;

V_ox_ullage_init = ullage_frac_init * V_ox_tank; % m^3
V_ox_init = V_ox_tank - V_ox_ullage_init; % m^3

A_reg_orifice = id_2_A(2.5e-3); % m^2

A_fuel_ullage_port = A_1_4_tube; % m^2
A_ox_ullage_port = A_1_4_tube; % m^2
% A_ox_ullage_port = A_3_8_tube; % m^2

A_fuel_tank_outlet = A_1_2_tube; % m^2
A_ox_tank_outlet = A_1_2_tube; % m^

A_pressurant_tank_outlet = A_1_4_tube; % m^2

fuel_feed_CdA = 14e-6; % m^2
ox_feed_CdA = 45e-6; % m^2

fuel_inj_CdA = 23.55e-6; % m^2
ox_inj_CdA = 79e-6; % m^2

fuel_inj_CdA = fuel_inj_CdA * 0.5; % m^2
ox_inj_CdA = ox_inj_CdA * 0.5; %

disp('Opening Simulink')

open_system('ereg_single_tank.slx');

function A = id_2_A(id)
    A = pi * (id/2)^2;
end

function m = in_2_m(in)
    m = in * 25.4e-3;
end