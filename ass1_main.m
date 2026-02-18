%% Assignment 1: 2-Class Classification
% Updated: Added visualization for Method 1 & 2 mistakes
clear;
clc;
close all;

%% Configuration
pairs = {{'chihuahua', 'muffin'}, ... % Hard Pair 1
         {'poodle', 'fried_chicken'}, ... % Hard Pair 2
         {'bus', 'truck'}}; % Easy Pair

baseDir = 'img';

%% Main Experiment Loop
for p_idx = 1:length(pairs)
    class1 = pairs{p_idx}{1};
    class2 = pairs{p_idx}{2};

    fprintf('\n========================================\n');
    fprintf('Starting Experiment: %s vs %s\n', class1, class2);
    fprintf('========================================\n');

    % 1. Data Loading
    dir1 = fullfile(baseDir, class1);
    dir2 = fullfile(baseDir, class2);

    if ~exist(dir1, 'dir') || ~exist(dir2, 'dir')
        warning('Data directories for %s vs %s not found. Skipping.', class1, class2);
        continue;
    end

    imds1 = imageDatastore(dir1, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imds2 = imageDatastore(dir2, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    numImages1 = length(imds1.Files);
    numImages2 = length(imds2.Files);
    minCount = min(numImages1, numImages2);

    if minCount < 50
        warning('Number of images is low (%d). Results may be unstable.', minCount);
    end
    fprintf('Using %d images per class (Total: %d)\n', minCount, minCount * 2);

    subimds1 = splitEachLabel(imds1, minCount, 'randomized');
    subimds2 = splitEachLabel(imds2, minCount, 'randomized');

    imds = imageDatastore(cat(1, subimds1.Files, subimds2.Files), ...
        'Labels', cat(1, subimds1.Labels, subimds2.Labels));

    cv = cvpartition(imds.Labels, 'KFold', 5);

    % Method 1 : Color Histogram + KNN
    fprintf('\n--- Method 1: Color Histogram + KNN ---\n');
    method1_acc = evaluate_color_hist(imds, cv);
    fprintf('Accuracy: %.2f%%\n', method1_acc * 100);

    % Method 2 : BoF + Nonlinear SVM
    fprintf('\n--- Method 2: Bag of Features (Dense) + Nonlinear SVM ---\n');
    method2_acc = evaluate_bof(imds, cv);
    fprintf('Accuracy: %.2f%%\n', method2_acc * 100);

    %% Setup Network for Method 3 & 4
    if ~isempty(which('resnet50'))
        net = resnet50();
        layer = 'avg_pool';
        fprintf('Using ResNet50 for DCNN methods.\n');
    elseif ~isempty(which('alexnet'))
        net = alexnet();
        layer = 'fc7';
        fprintf('Using AlexNet for DCNN methods.\n');
    else
        error('No pretrained network found. Please install ResNet50 or AlexNet.');
    end

    % Method 3 : DCNN Features + Linear SVM
    fprintf('\n--- Method 3: DCNN (ResNet50) + Linear SVM ---\n');
    method3_acc = evaluate_dcnn(imds, cv, net, layer);
    fprintf('Accuracy: %.2f%%\n', method3_acc * 100);

    % Method 4 : DCNN Features + Non-linear SVM (Advanced)
    fprintf('\n--- Method 4: DCNN (ResNet50) + Non-linear SVM (RBF) ---\n');
    method4_acc = evaluate_dcnn_nonlinear(imds, cv, net, layer);
    fprintf('Accuracy: %.2f%%\n', method4_acc * 100);

    %% Summary for this Pair
    fprintf('\n=== Results Summary: %s vs %s ===\n', class1, class2);
    fprintf('Method 1 (Color+KNN):      %.2f%%\n', method1_acc * 100);
    fprintf('Method 2 (BoF+SVM):        %.2f%%\n', method2_acc * 100);
    fprintf('Method 3 (DCNN+Linear):    %.2f%%\n', method3_acc * 100);
    fprintf('Method 4 (DCNN+RBF):       %.2f%% (Advanced)\n', method4_acc * 100);
    fprintf('===========================================\n');
end

%% Local Functions

function mean_acc = evaluate_color_hist(imds, cv)
    imds.ReadFcn = @readHsvHist;
    features = cell2mat(imds.readall());
    imds.ReadFcn = @safeImread;

    accuracies = zeros(cv.NumTestSets, 1);
    for i = 1 : cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        XTrain = features(trainIdx, :);
        YTrain = imds.Labels(trainIdx);
        XTest = features(testIdx, :);
        YTest = imds.Labels(testIdx);

        mdl = fitcknn(XTrain, YTrain, 'NumNeighbors', 1);
        YPred = predict(mdl, XTest);
        accuracies(i) = sum(YPred == YTest) / length(YTest);
    end
    mean_acc = mean(accuracies);
    visualize_mistakes(imds, YTest, YPred, testIdx, 'Method1_ColorKNN');
end

function histVec = readHsvHist(filename)
    try
        I = imread(filename);
        I = ensureRGB(I);
        hsv = rgb2hsv(I);
        H = floor(hsv(:, :, 1) * 3.99);
        S = floor(hsv(:, :, 2) * 3.99);
        V = floor(hsv(:, :, 3) * 3.99);
        idx = H * 16 + S * 4 + V;
        histVec = histcounts(idx, 0:64, 'Normalization', 'pdf');
    catch
        [~, name, ext] = fileparts(filename);
        warning('Skipping corrupt image in Color Hist: %s%s', name, ext);
        histVec = zeros(1, 64);
    end
end

function mean_acc = evaluate_bof(imds, cv)
    imds.ReadFcn = @safeImread;
    extractor = @denseSurfExtractor;

    fprintf('Creating Bag of Features...\n');
    bag = bagOfFeatures(imds, 'CustomExtractor', extractor, ...
        'VocabularySize', 1000, 'StrongestFeatures', 1.0);

    fprintf('Encoding images...\n');
    features = encode(bag, imds);

    accuracies = zeros(cv.NumTestSets, 1);
    for i = 1 : cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        XTrain = features(trainIdx, :);
        YTrain = imds.Labels(trainIdx);
        XTest = features(testIdx, :);
        YTest = imds.Labels(testIdx);

        t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
        mdl = fitcecoc(XTrain, YTrain, 'Learners', t);
        YPred = predict(mdl, XTest);
        accuracies(i) = sum(YPred == YTest) / length(YTest);
    end
    mean_acc = mean(accuracies);
    visualize_mistakes(imds, YTest, YPred, testIdx, 'Method2_BoF');
end

function I = safeImread(filename)
    try
        I = imread(filename);
        I = ensureRGB(I);
    catch
        warning('Skipping corrupt image in BoF: %s', filename);
        I = zeros(300, 300, 3, 'uint8');
    end
end

% === Custom Extractor using provided createRandomPoints ===
function [features, metrics] = denseSurfExtractor(I)
    try
        I = ensureRGB(I);
        I_gray = rgb2gray(I);
        
        % Use the requested custom function
        points = createRandomPoints(I_gray, 1000);
        
        [features, validPoints] = extractFeatures(I_gray, points);
        
        % FIX: Handle empty features to prevent bagOfFeatures validation error
        if isempty(features)
            features = zeros(1, 64, 'single');
            metrics = 0;
        else
            if isa(validPoints, 'SURFPoints') || isa(validPoints, 'MSERRegions') || isa(validPoints, 'cornerPoints')
                metrics = validPoints.Metric;
            else
                metrics = ones(size(features, 1), 1, 'single');
            end
        end
    catch
        % Fallback for corrupt images
        features = zeros(1, 64, 'single');
        metrics = 0;
    end
end

% === Requested Function ===
function PT = createRandomPoints(I, num)
    [sy, sx] = size(I);
    sz = [sx sy];
    for i = 1:num
        s = 0;
        while s < 1.6
            s = randn() * 3 + 3;
        end
        p = ceil((sz - ceil(s) * 2) .* rand(1, 2) + ceil(s));
        if i == 1
            PT = [SURFPoints(p, 'Scale', s)];
        else
            PT = [PT; SURFPoints(p, 'Scale', s)];
        end
    end
end

function mean_acc = evaluate_dcnn(imds, cv, net, layer)
    inputSize = net.Layers(1).InputSize;
    imds.ReadFcn = @(f) readForDCNN(f, inputSize);

    fprintf('Extracting DCNN features...\n');
    features = activations(net, imds, layer, 'OutputAs', 'rows');

    accuracies = zeros(cv.NumTestSets, 1);
    for i = 1 : cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        XTrain = features(trainIdx, :);
        YTrain = imds.Labels(trainIdx);
        XTest = features(testIdx, :);
        YTest = imds.Labels(testIdx);

        t = templateSVM('KernelFunction', 'linear', 'Solver', 'ISDA');
        mdl = fitcecoc(XTrain, YTrain, 'Learners', t);
        YPred = predict(mdl, XTest);
        accuracies(i) = sum(YPred == YTest) / length(YTest);
    end
    mean_acc = mean(accuracies);
    visualize_mistakes(imds, YTest, YPred, testIdx, 'Method3_DCNN_Linear');
end

function mean_acc = evaluate_dcnn_nonlinear(imds, cv, net, layer)
    inputSize = net.Layers(1).InputSize;
    imds.ReadFcn = @(f) readForDCNN(f, inputSize);

    fprintf('Extracting DCNN features (Method 4)...\n');
    features = activations(net, imds, layer, 'OutputAs', 'rows');

    accuracies = zeros(cv.NumTestSets, 1);
    for i = 1 : cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        XTrain = features(trainIdx, :);
        YTrain = imds.Labels(trainIdx);
        XTest = features(testIdx, :);
        YTest = imds.Labels(testIdx);

        t = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
        mdl = fitcecoc(XTrain, YTrain, 'Learners', t);
        YPred = predict(mdl, XTest);
        accuracies(i) = sum(YPred == YTest) / length(YTest);
    end
    mean_acc = mean(accuracies);
    visualize_mistakes(imds, YTest, YPred, testIdx, 'Method4_DCNN_RBF');
end

function visualize_mistakes(imds, YTest, YPred, testIdx, methodTag)
    wrongIdx = find(YPred ~= YTest);
    if ~isempty(wrongIdx)
        fprintf('Visualizing misclassified images (%s)...\n', methodTag);
        currentTestFiles = imds.Files(testIdx);
        wrongFiles = currentTestFiles(wrongIdx);
        
        thumbnails = cell(1, length(wrongFiles));
        for k = 1 : length(wrongFiles)
            try
                img = imread(wrongFiles{k});
                if size(img, 3) == 1
                    img = cat(3, img, img, img);
                end
                thumbnails{k} = img;
            catch
                thumbnails{k} = zeros(150, 150, 3, 'uint8');
            end
        end
        
        figure('Name', ['Mistakes ' methodTag]);
        montage(thumbnails, 'ThumbnailSize', [150 150]);
        title(sprintf('%s Mistakes (Total: %d)', methodTag, length(wrongFiles)));
        
        timestamp = datestr(now, 'yyyyMMdd_HHmmss');
        saveas(gcf, sprintf('ass1_Mistakes_%s_%s.png', methodTag, timestamp));
    else
        fprintf('No mistakes found in the last fold (%s)!\n', methodTag);
    end
end

function I_out = readForDCNN(filename, inputSize)
    try
        I = imread(filename);
        I = ensureRGB(I);
        I_out = imresize(I, inputSize(1:2));
    catch
        [~, name, ext] = fileparts(filename);
        warning('Skipping corrupt image in DCNN: %s%s', name, ext);
        I_out = zeros([inputSize(1:2), 3], 'uint8');
    end
end

function I_out = ensureRGB(I)
    if size(I, 3) == 1
        I_out = cat(3, I, I, I);
    else
        I_out = I;
    end
end