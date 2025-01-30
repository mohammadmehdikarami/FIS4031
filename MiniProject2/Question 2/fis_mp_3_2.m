clc;
clear;
close all;

%% Create Fuzzy Inference System
fis_name = 'TruckController';
fis_type = 'mamdani';
and_method = 'prod';
or_method = 'max';
imp_method = 'prod';
agg_method = 'max';
defuzz_method = 'centroid';
fis = newfis(fis_name, fis_type, and_method, or_method, imp_method, agg_method, defuzz_method);

%% Define Inputs and Output
fis = addvar(fis, 'input', 'x', [0 20]);
fis = addvar(fis, 'input', 'phi', [-90 270]);
fis = addvar(fis, 'output', 'theta', [-40 40]);

%% Define Membership Functions for Inputs
fis = addmf(fis, 'input', 1, 'S2', 'trapmf', [0 0 1.5 7]);
fis = addmf(fis, 'input', 1, 'S1', 'trimf', [4 7 10]);
fis = addmf(fis, 'input', 1, 'CE', 'trimf', [9 10 11]);
fis = addmf(fis, 'input', 1, 'B1', 'trimf', [10 13 16]);
fis = addmf(fis, 'input', 1, 'B2', 'trapmf', [13 18.5 20 20]);

fis = addmf(fis, 'input', 2, 'S3', 'trimf', [-115 -65 -15]);
fis = addmf(fis, 'input', 2, 'S2', 'trimf', [-45 0 45]);
fis = addmf(fis, 'input', 2, 'S1', 'trimf', [15 52.5 90]);
fis = addmf(fis, 'input', 2, 'CE', 'trimf', [80 90 100]);
fis = addmf(fis, 'input', 2, 'B1', 'trimf', [90 127.5 165]);
fis = addmf(fis, 'input', 2, 'B2', 'trimf', [135 180 225]);
fis = addmf(fis, 'input', 2, 'B3', 'trimf', [180 225 295]);

%% Define Membership Functions for Output
fis = addmf(fis, 'output', 1, 'S3', 'trimf', [-60 -40 -20]);
fis = addmf(fis, 'output', 1, 'S2', 'trimf', [-33 -20 -7]);
fis = addmf(fis, 'output', 1, 'S1', 'trimf', [-14 -7 0]);
fis = addmf(fis, 'output', 1, 'CE', 'trimf', [-4 0 4]);
fis = addmf(fis, 'output', 1, 'B1', 'trimf', [0 7 14]);
fis = addmf(fis, 'output', 1, 'B2', 'trimf', [7 20 33]);
fis = addmf(fis, 'output', 1, 'B3', 'trimf', [20 40 60]);

%% Define Fuzzy Rules
rules = [...
    1 1 2 1 1; 1 2 2 1 1; 1 3 5 1 1; 1 4 6 1 1; 1 5 6 1 1; ...
    2 1 1 1 1; 2 2 1 1 1; 2 3 3 1 1; 2 4 6 1 1; 2 5 7 1 1; ...
    3 2 1 1 1; 3 3 2 1 1; 3 4 4 1 1; 3 5 6 1 1; 3 6 7 1 1; ...
    4 2 1 1 1; 4 3 1 1 1; 4 4 2 1 1; 4 5 5 1 1; 4 6 7 1 1; ...
    5 3 2 1 1; 5 4 2 1 1; 5 5 3 1 1; 5 6 6 1 1; 5 7 6 1 1];
fis = addrule(fis, rules);

%% Visualization
figure;
plotmf(fis, 'input', 1);
title('Membership Functions for Input x');
figure;
plotmf(fis, 'input', 2);
title('Membership Functions for Input phi');
figure;
plotmf(fis, 'output', 1);
title('Membership Functions for Output theta');
figure;
gensurf(fis);
title('FIS Output Surface');

%% Truck Control Simulation
b = 4;
n = 250;
trajectory = zeros(n, 5);

x = zeros(1, n);
phi = zeros(1, n);
y = zeros(1, n);
y(1) = 2;

x(1) = input('Enter initial x (0 < x < 20): ');
phi(1) = input('Enter initial phi (-90 < phi < 270): ');

desired_x = 10;
desired_phi = 90;

cost = norm([desired_x - x(1), desired_phi - phi(1)]);
t = 1;

while cost >= 0.01
    theta = evalfis([x(t); phi(t)], fis);
    trajectory(t, :) = [t-1, x(t), y(t), phi(t), theta];
    x(t+1) = x(t) + cosd(phi(t) + theta) + sind(theta) * sind(phi(t));
    phi(t+1) = phi(t) - asind(2 * sind(theta) / b);
    y(t+1) = y(t) + sind(phi(t) + theta) - sind(theta) * cosd(phi(t));
    cost = norm([desired_x - x(t+1), desired_phi - phi(t+1)]);
    t = t + 1;
end

x_traj = x(1:t);
y_traj = y(1:t);

fprintf('Final Position: x = %.2f, y = %.2f, phi = %.2f\n', x_traj(end), y_traj(end), phi(t));

figure;
plot(x_traj, y_traj, 'LineWidth', 2);
xlabel('x');
ylabel('y');
title('Truck Trajectory');
grid on;