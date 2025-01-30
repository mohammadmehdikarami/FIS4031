clc;
clear all;
close all;

% Load the data from the Excel file
data = readtable('AirQualityUCI.xlsx');

% Extract the NO2(GT) column as the output
outputData = data.NO2_GT_;

% Extract the input features (all columns except NO2(GT), Date, and Time)
inputData = data{:, setdiff(data.Properties.VariableNames, {'NO2_GT_', 'Date', 'Time'})};

% Handle missing values (replace -200 with NaN)
inputData(inputData == -200) = NaN;
outputData(outputData == -200) = NaN;

% Remove rows with missing values
validRows = ~any(isnan(inputData), 2) & ~isnan(outputData);
inputData = inputData(validRows, :);
outputData = outputData(validRows, :);

% Manually calculate mean (mu) and standard deviation (sigma) for input and output data
inputMu = mean(inputData, 1);  % Mean of each column (1x12 array)
inputSigma = std(inputData, 0, 1);  % Standard deviation of each column (1x12 array)

outputMu = mean(outputData, 1);  % Mean of output data (scalar)
outputSigma = std(outputData, 0, 1);  % Standard deviation of output data (scalar)

% Normalize the input and output data using MATLAB's normalize function
inputDataNorm = normalize(inputData);  % Normalize input data
outputDataNorm = normalize(outputData);  % Normalize output data

% Split the data into training, testing, and validation sets
rng(1);  % For reproducibility
n = size(inputDataNorm, 1);
trainIndices = 1:round(0.6*n);
testIndices = round(0.6*n)+1:round(0.8*n);
valIndices = round(0.8*n)+1:n;

trainInput = inputDataNorm(trainIndices, :);
trainOutput = outputDataNorm(trainIndices, :);

testInput = inputDataNorm(testIndices, :);
testOutput = outputDataNorm(testIndices, :);

valInput = inputDataNorm(valIndices, :);
valOutput = outputDataNorm(valIndices, :);

% Generate the initial FIS structure using genfis2
radius = 0.5;  % Adjust this parameter as needed
in_fis = genfis2(trainInput, trainOutput, radius);

% Train the ANFIS model and capture training/checking error
epochs = 100;  % Number of epochs
[out_fis, trainError, stepSize,~, valError] = anfis([trainInput, trainOutput], in_fis, epochs, [1, 1, 1, 1], [valInput, valOutput]);

% Evaluate the model on the test set
predictedOutputNorm = evalfis(testInput, out_fis);

% Denormalize the predicted and actual outputs for the test set
predictedOutputDenorm = (predictedOutputNorm * outputSigma) + outputMu;
testOutputDenorm = (testOutput * outputSigma) + outputMu;

% Define the RBF neural network
numRBFNeurons = 20;
net = newrb(trainInput', trainOutput', 0, 5, numRBFNeurons, 1);

% Evaluate the RBF network on the test set
predictedOutputRBF = sim(net, testInput');

% Calculate RMSE and MSE for ANFIS and RBF models
testErrorDenormANFIS = predictedOutputDenorm - testOutputDenorm;
testRMSE_ANFIS = sqrt(mean(testErrorDenormANFIS.^2));
testMSE_ANFIS = mean(testErrorDenormANFIS.^2);

testErrorDenormRBF = predictedOutputRBF - testOutputDenorm';
testRMSE_RBF = sqrt(mean(testErrorDenormRBF.^2));
testMSE_RBF = mean(testErrorDenormRBF.^2);

% Report denormalized RMSE and MSE for both models
fprintf('ANFIS Model:\n');
fprintf('  Test RMSE: %.4f\n', testRMSE_ANFIS);
fprintf('  Test MSE: %.4f\n', testMSE_ANFIS);

fprintf('RBF Model:\n');
fprintf('  Test RMSE: %.4f\n', testRMSE_RBF);
fprintf('  Test MSE: %.4f\n', testMSE_RBF);

% Plot the test set results for both models
figure;
plot(testOutputDenorm, 'b');
hold on;
plot(predictedOutputDenorm, 'r');
plot(predictedOutputRBF, 'g');
legend('Actual Output', 'Predicted Output (ANFIS)', 'Predicted Output (RBF)');
xlabel('Sample Index');
ylabel('NO2(GT)');
title('Test Set: Actual vs Predicted (ANFIS & RBF)');
hold off;