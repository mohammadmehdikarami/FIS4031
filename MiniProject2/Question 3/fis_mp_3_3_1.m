%% Load and Preprocess Data
clear; clc;

% Load dataset
rawData = load("ballbeam.dat");

% Extract input and target
numSamples = size(rawData, 1);
inputs = rawData(:, 1);
targets = rawData(:, 2);

% Plot the data
figure;
plot(inputs, targets, 'b.');
title('Input vs Target Data');
xlabel('Input');
ylabel('Target');
grid on;

% Set random seed for reproducibility
rng(13); % Based on student ID: 40003913

% Shuffle indices
shuffledIdx = randperm(numSamples);

% Define data split ratios
trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

% Compute the number of samples in each set
numTrain = round(trainRatio * numSamples);
numVal = round(valRatio * numSamples);
numTest = numSamples - (numTrain + numVal); % Ensure correct total count

% Assign indices to sets
trainIdx = shuffledIdx(1:numTrain);
valIdx = shuffledIdx(numTrain+1:numTrain+numVal);
testIdx = shuffledIdx(numTrain+numVal+1:end);

% Split data
trainData = rawData(trainIdx, :);
valData = rawData(valIdx, :);
testData = rawData(testIdx, :);

% Extract input and target for each subset
TrainInputs = trainData(:, 1);
TrainTargets = trainData(:, 2);
ValInputs = valData(:, 1);
ValTargets = valData(:, 2);
TestInputs = testData(:, 1);
TestTargets = testData(:, 2);

%% ANFIS Training
numModels = 50; % Number of models to train
TestRMSE = zeros(numModels, 1);
fis_models = cell(numModels, 1);
TrainRMSE = cell(numModels, 1);
ValRMSE = cell(numModels, 1);

for i = 1:numModels
    % Generate FIS structure using genfis2
    radius = (i + 5) / 100;
    fis_models{i} = genfis2(TrainInputs, TrainTargets, radius);
    
    % Check the number of rules in the generated FIS
    numRules = length(fis_models{i}.rule);
    if numRules < 2
        warning("Generated FIS has only %d rule(s). Consider adjusting the radius.", numRules);
        continue; % Skip training if there aren't enough rules
    end
    
    % Training options
    MaxEpoch = 1000;
    ErrorGoal = 0;
    InitialStepSize = 0.01;
    StepSizeDecreaseRate = 0.999;
    StepSizeIncreaseRate = 1.001;
    TrainOptions = [MaxEpoch, ErrorGoal, InitialStepSize, StepSizeDecreaseRate, StepSizeIncreaseRate];
    
    % Display settings
    DisplayOptions = [true, false, false, true];
    
    % Train ANFIS model
    [~, TrainRMSE{i}, ~, fis_models{i}, ValRMSE{i}] = anfis(trainData, fis_models{i}, TrainOptions, DisplayOptions, valData, 1);
    
    % Evaluate on test set
    TestOutputs = evalfis(fis_models{i}, TestInputs);
    TestErrors = TestTargets - TestOutputs;
    TestMSE = mean(TestErrors.^2);
    TestRMSE(i) = sqrt(TestMSE);
end

% Convert RMSE results to matrices
TrainRMSE = cell2mat(TrainRMSE);
ValRMSE = cell2mat(ValRMSE);
TrainRMSE = min(TrainRMSE);
ValRMSE = min(ValRMSE);

% Plot RMSE trends
figure;
plot(TestRMSE, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
title('Test RMSE for Different ANFIS Models');
xlabel('Model Index');
ylabel('Test RMSE');
grid on;

% Select best-performing model
[~, bestModelIdx] = min(TestRMSE);
bestFIS = fis_models{bestModelIdx};

%% Evaluate Best Model on Different Sets
% Train Set Evaluation
TrainOutputs = evalfis(bestFIS, TrainInputs);
TrainErrors = TrainTargets - TrainOutputs;
TrainRMSE = sqrt(mean(TrainErrors.^2));

% Validation Set Evaluation
ValOutputs = evalfis(bestFIS, ValInputs);
ValErrors = ValTargets - ValOutputs;
ValRMSE = sqrt(mean(ValErrors.^2));

% Test Set Evaluation
TestOutputs = evalfis(bestFIS, TestInputs);
TestErrors = TestTargets - TestOutputs;
TestRMSE = sqrt(mean(TestErrors.^2));

% Overall Dataset Evaluation
AllOutputs = evalfis(bestFIS, inputs);
AllErrors = targets - AllOutputs;
AllRMSE = sqrt(mean(AllErrors.^2));

% Plot results
figure;
plot(targets, AllOutputs, 'ro');
title('ANFIS Model Predictions vs Actual Targets');
xlabel('Actual Target');
ylabel('Predicted Output');
grid on;
