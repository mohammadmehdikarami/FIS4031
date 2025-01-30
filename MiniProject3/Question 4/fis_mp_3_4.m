% MATLAB Code for Offline Identification of f(u(k)) using ANFIS
clear;
clc;

% Generate training input signal (sinusoidal)
k_max = 1000; 
u_train = sin(2*pi*(1:k_max)/250); 

% Define the unknown nonlinear function f(u)
f = @(u) 0.6*sin(pi*u) + 0.3*sin(3*pi*u) + 0.1*sin(5*pi*u);

% Generate plant output y(k) based on the dynamics
y_train = zeros(1, k_max); 
y_train(1) = 0; 
y_train(2) = 0; 
for k = 2:k_max-1
    y_train(k+1) = 0.3*y_train(k) + 0.6*y_train(k-1) + f(u_train(k)); 
end

% Prepare data for ANFIS training: [u(k), f(u(k))]
f_u_train = zeros(1, k_max-2); 
for k = 2:k_max-1
    f_u_train(k-1) = y_train(k+1) - 0.3*y_train(k) - 0.6*y_train(k-1); 
end
input_data_train = u_train(2:k_max-1)'; 
output_data_train = f_u_train'; 

% Initialize ANFIS with grid partition method
num_mf = 7; 
genfis_opt = genfisOptions('GridPartition', 'NumMembershipFunctions', num_mf);
fis = genfis(input_data_train, output_data_train, genfis_opt); 

% Display initial membership functions
figure;
plotmf(fis, 'input', 1);
title('Initial Membership Functions of Input u(k)');

% Set training options for ANFIS
opt = anfisOptions('InitialFIS', fis, ...
                   'EpochNumber', 200, ... 
                   'InitialStepSize', 0.1, ...
                   'StepSizeDecreaseRate', 0.9, ...
                   'StepSizeIncreaseRate', 1.1, ...
                   'DisplayANFISInformation', 1, ...
                   'DisplayErrorValues', 1, ...
                   'DisplayStepSize', 1, ...
                   'DisplayFinalResults', 1);

% Train ANFIS and capture training error
[fis, trainError] = anfis([input_data_train, output_data_train], opt);

% Predict f(u(k)) for training data using trained ANFIS
f_u_train_hat = evalfis(fis, input_data_train);

% Simulate plant output using the identified f(u(k))
y_train_hat = zeros(1, k_max); 
y_train_hat(1) = 0; 
y_train_hat(2) = 0; 
for k = 2:k_max-1
    y_train_hat(k+1) = 0.3*y_train_hat(k) + 0.6*y_train_hat(k-1) + f_u_train_hat(k-1); 
end

% Plot training results: Actual vs. Predicted Output
figure;
subplot(2,1,1);
plot(1:k_max, y_train, 'b', 'LineWidth', 1.5); hold on;
plot(1:k_max, y_train_hat, 'r--', 'LineWidth', 1.5);
legend('Actual Output (y)', 'Predicted Output (y\_hat)');
xlabel('Time Step (k)');
ylabel('Output');
title('Training Phase: Actual vs. Predicted Output');

% Plot prediction error for training phase
subplot(2,1,2);
plot(1:k_max, y_train - y_train_hat, 'g', 'LineWidth', 1.5);
legend('Prediction Error');
xlabel('Time Step (k)');
ylabel('Error');
title('Training Phase: Prediction Error');

% Plot training error (loss) over epochs
figure;
plot(trainError, 'LineWidth', 1.5);
xlabel('Epochs');
ylabel('Training Error (Loss)');
title('Training Error vs. Epochs');

% Visualize ANFIS network architecture
figure;
plotfis(fis);
title('ANFIS Network Architecture');

% Display final membership functions of input after training
figure;
plotmf(fis, 'input', 1);
title('Final Membership Functions of Input u(k)');

% Test the trained ANFIS with new input signal
k_max_test = 1000; 
u_test = 0.5*sin(2*pi*(1:k_max_test)/250) + 0.5*sin(2*pi*(1:k_max_test)/25); 

% Generate plant output for the test input
y_test = zeros(1, k_max_test); 
y_test(1) = 0; 
y_test(2) = 0; 
for k = 2:k_max_test-1
    y_test(k+1) = 0.3*y_test(k) + 0.6*y_test(k-1) + f(u_test(k)); 
end

% Predict f(u(k)) for the test data using trained ANFIS
f_u_test_hat = evalfis(fis, u_test(2:k_max_test-1)');

% Simulate plant output using identified f(u(k)) for test input
y_test_hat = zeros(1, k_max_test); 
y_test_hat(1) = 0; 
y_test_hat(2) = 0; 
for k = 2:k_max_test-1
    y_test_hat(k+1) = 0.3*y_test_hat(k) + 0.6*y_test_hat(k-1) + f_u_test_hat(k-1); 
end

% Plot testing results: Actual vs. Predicted Output
figure;
subplot(2,1,1);
plot(1:k_max_test, y_test, 'b', 'LineWidth', 1.5); hold on;
plot(1:k_max_test, y_test_hat, 'r--', 'LineWidth', 1.5);
legend('Actual Output (y)', 'Predicted Output (y\_hat)');
xlabel('Time Step (k)');
ylabel('Output');
title('Testing Phase: Actual vs. Predicted Output');

% Plot prediction error for testing phase
subplot(2,1,2);
plot(1:k_max_test, y_test - y_test_hat, 'g', 'LineWidth', 1.5);
legend('Prediction Error');
xlabel('Time Step (k)');
ylabel('Error');
title('Testing Phase: Prediction Error');
