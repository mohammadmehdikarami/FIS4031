rng(13); 
clusterRadius = 0.4;
maxEpochs = 100;
errorGoal = 0;
initStepSize = 0.01;
stepSizeDecr = 0.99;
stepSizeIncr = 1.01;

trainOptions = [maxEpochs, errorGoal, initStepSize, stepSizeDecr, stepSizeIncr];
dispOptions = [1, 0, 0, 1];
optMethod = 1;

rawData = load('steamgen.dat');

if size(rawData, 2) < 9
    error('Data must have at least 9 columns: time + 4 inputs + 4 outputs.');
end

inputs_raw = rawData(:, 2:5);
outputs_raw = rawData(:, 6:9);

figure;
plot(rawData(:, 1), rawData(:, 2));
title('First Input vs. Time');
xlabel('Time index');
ylabel('Fuel Input');

[inputs_norm, inMin, inMax] = normalizeMinMax(inputs_raw);
[outputs_norm, outMin, outMax] = normalizeMinMax(outputs_raw);

dataNorm = [inputs_norm, outputs_norm];
N = size(dataNorm, 1);

trainRatio = 0.70;
valRatio = 0.15;
testRatio = 0.15;

if abs(trainRatio + valRatio + testRatio - 1.0) > 1e-9
    error('Train/Validation/Test split ratios must sum to 1.0.');
end

randIdx = randperm(N);
nTrain = round(trainRatio * N);
nVal = round(valRatio * N);
nTest = N - nTrain - nVal;

idxTrain = randIdx(1:nTrain);
idxVal = randIdx(nTrain+1:nTrain+nVal);
idxTest = randIdx(nTrain+nVal+1:end);

trainData = dataNorm(idxTrain, :);
valData = dataNorm(idxVal, :);
testData = dataNorm(idxTest, :);

X_train = trainData(:, 1:4);
Y_train = trainData(:, 5:8);
X_val = valData(:, 1:4);
Y_val = valData(:, 5:8);
X_test = testData(:, 1:4);
Y_test = testData(:, 5:8);

fisList = cell(1, 4);
trainErrorList = cell(1, 4);
valErrorList = cell(1, 4);

for outIdx = 1:4
    trainDataSingle = [X_train, Y_train(:, outIdx)];
    initFIS = genfis2(X_train, Y_train(:, outIdx), clusterRadius);
    [fisTrained, trainError, ~, fisFinal, valError] = anfis(trainDataSingle, initFIS, trainOptions, dispOptions, [X_val, Y_val(:, outIdx)], optMethod);
    fisList{outIdx} = fisFinal;
    trainErrorList{outIdx} = trainError;
    valErrorList{outIdx} = valError;
end

rmseTrain = zeros(1, 4);
rmseVal = zeros(1, 4);
rmseTest = zeros(1, 4);

for outIdx = 1:4
    yhat_train = evalfis(fisList{outIdx}, X_train);
    rmseTrain(outIdx) = sqrt(mean((Y_train(:, outIdx) - yhat_train).^2));
    yhat_val = evalfis(fisList{outIdx}, X_val);
    rmseVal(outIdx) = sqrt(mean((Y_val(:, outIdx) - yhat_val).^2));
    yhat_test = evalfis(fisList{outIdx}, X_test);
    rmseTest(outIdx) = sqrt(mean((Y_test(:, outIdx) - yhat_test).^2));
end

disp('RMSE Results:');
disp(table((1:4)', rmseTrain', rmseVal', rmseTest', 'VariableNames', {'Output', 'TrainRMSE', 'ValRMSE', 'TestRMSE'}));

figure('Name', 'ANFIS Training and Validation Errors');
for outIdx = 1:4
    subplot(2, 2, outIdx);
    plot(trainErrorList{outIdx}, 'LineWidth', 1.5); hold on;
    plot(valErrorList{outIdx}, 'LineWidth', 1.5);
    title(['Output #', num2str(outIdx), ' - Learning Curve']);
    xlabel('Epoch'); ylabel('RMSE');
    legend('TrainError', 'ValError'); grid on;
end

figure('Name', 'Outputs vs Targets (Test Set)');
for outIdx = 1:4
    subplot(2, 2, outIdx);
    yhat_test = evalfis(fisList{outIdx}, X_test);
    plot(Y_test(:, outIdx), 'b', 'LineWidth', 1); hold on;
    plot(yhat_test, 'r', 'LineWidth', 1);
    title(['Output #', num2str(outIdx), ' on Test Data']);
    xlabel('Sample'); ylabel('Normalized Value');
    legend('Target', 'Predicted'); grid on;
end

figure;
for i = 1:4
    for j = 1:4
        subplot(4, 4, j + ((i - 1) * 4));
        plotmf(fisList{i}, 'input', j);
        title(['MFs of input ', num2str(j), ' for Output #', num2str(i), ' FIS']);
    end
end

for i = 1:4
    figure; plotfis(fisList{i});
    title(['FIS Structure for Output #', num2str(i)]);
end

disp('ANFIS Multi-Output Modeling Completed Successfully!');

function [dataNorm, dataMin, dataMax] = normalizeMinMax(data)
    dataMin = min(data, [], 1);
    dataMax = max(data, [], 1);
    dataRange = dataMax - dataMin;
    dataRange(dataRange == 0) = 1e-12;
    dataNorm = (data - dataMin) ./ dataRange;
end
