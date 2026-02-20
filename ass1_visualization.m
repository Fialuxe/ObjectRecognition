%% Assignment 1: 2-Class Classification with Visualization (t-SNE & Grad-CAM)

function assignment1_with_viz()
    clc; clear; close all;

    %% 1. Configuration
    Config = struct();
    Config.BaseDir = 'img';
    Config.ClassPairs = {
        {'chihuahua', 'muffin'}, ... % Hard Pair 1
        {'poodle', 'fried_chicken'}, ... % Hard Pair 2
        {'bus', 'truck'}             ... % Easy Pair
    };

    %% 2. Deep Learning Network Initialization
    [deepNet, featureLayer] = initialize_deep_network();

    %% 3. Main Experiment Loop
    for pairIdx = 1:length(Config.ClassPairs)
        class1 = Config.ClassPairs{pairIdx}{1};
        class2 = Config.ClassPairs{pairIdx}{2};
        pairName = sprintf('%s_vs_%s', class1, class2);
        
        print_section_header(sprintf('Starting Experiment: %s', pairName));

        % --- Data Loading & Balancing ---
        [balancedImageStore, minCount] = load_and_balance_data(Config.BaseDir, class1, class2);
        if isempty(balancedImageStore)
            continue;
        end
        fprintf('Using %d images per class (Total: %d)\n', minCount, minCount * 2);

        % --- Setup Cross Validation ---
        cvPartition = cvpartition(balancedImageStore.Labels, 'KFold', 5);

        % --- Method 1: Color Histogram + KNN ---
        fprintf('\n--- Method 1: Color Histogram + KNN ---\n');
        evaluate_color_histogram(balancedImageStore, cvPartition, pairName);

        % --- Method 2: BoF + Nonlinear SVM ---
        fprintf('\n--- Method 2: Bag of Features (Dense) + Nonlinear SVM ---\n');
        evaluate_bag_of_features(balancedImageStore, cvPartition, pairName);

        % --- Method 3 & 4: DCNN Features ---
        fprintf('\n--- Extracting DCNN Features ---\n');
        dcnnFeatures = extract_dcnn_features(balancedImageStore, deepNet, featureLayer);
        
        % DCNN features are identical for Method 3 and 4, so we plot t-SNE once.
        visualize_feature_space(dcnnFeatures, balancedImageStore.Labels, ['DCNN_Features_' pairName]);

        fprintf('\n--- Method 3: DCNN + Linear SVM ---\n');
        evaluate_dcnn_model(balancedImageStore, cvPartition, dcnnFeatures, 'linear', 'Method3_DCNN_Linear', deepNet);

        fprintf('\n--- Method 4: DCNN + Non-linear SVM (RBF) ---\n');
        evaluate_dcnn_model(balancedImageStore, cvPartition, dcnnFeatures, 'rbf', 'Method4_DCNN_RBF', deepNet);
    end
end

%% ========================================================================
%% Domain Logic Functions (Evaluators)
%% ========================================================================

function evaluate_color_histogram(imageStore, cvPartition, pairName)
    imageStore.ReadFcn = @read_hsv_histogram;
    features = cell2mat(imageStore.readall());
    imageStore.ReadFcn = @read_safe_image; % Reset
    
    % Visualize Feature Space
    visualize_feature_space(features, imageStore.Labels, ['Method1_ColorHist_' pairName]);

    trainPredictFn = @(XTrain, YTrain, XTest) predict(fitcknn(XTrain, YTrain, 'NumNeighbors', 1), XTest);
    [meanAccuracy, lastYTest, lastYPred, lastTestIdx] = execute_cross_validation(features, imageStore.Labels, cvPartition, trainPredictFn);
    
    fprintf('Accuracy: %.2f%%\n', meanAccuracy * 100);
    visualize_mistakes(imageStore, lastYTest, lastYPred, lastTestIdx, 'Method1_ColorKNN');
end

function evaluate_bag_of_features(imageStore, cvPartition, pairName)
    imageStore.ReadFcn = @read_safe_image;
    
    fprintf('Creating Bag of Features...\n');
    bag = bagOfFeatures(imageStore, 'CustomExtractor', @extract_dense_surf, 'VocabularySize', 1000, 'StrongestFeatures', 1.0);
    features = encode(bag, imageStore);

    % Visualize Feature Space
    visualize_feature_space(features, imageStore.Labels, ['Method2_BoF_' pairName]);

    svmTemplate = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
    trainPredictFn = @(XTrain, YTrain, XTest) predict(fitcecoc(XTrain, YTrain, 'Learners', svmTemplate), XTest);

    [meanAccuracy, lastYTest, lastYPred, lastTestIdx] = execute_cross_validation(features, imageStore.Labels, cvPartition, trainPredictFn);
    
    fprintf('Accuracy: %.2f%%\n', meanAccuracy * 100);
    visualize_mistakes(imageStore, lastYTest, lastYPred, lastTestIdx, 'Method2_BoF');
end

function evaluate_dcnn_model(imageStore, cvPartition, features, kernelType, methodTag, deepNet)
    if strcmp(kernelType, 'linear')
        svmTemplate = templateSVM('KernelFunction', 'linear', 'Solver', 'ISDA');
    else
        svmTemplate = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
    end
    
    trainPredictFn = @(XTrain, YTrain, XTest) predict(fitcecoc(XTrain, YTrain, 'Learners', svmTemplate), XTest);
    [meanAccuracy, lastYTest, lastYPred, lastTestIdx] = execute_cross_validation(features, imageStore.Labels, cvPartition, trainPredictFn);
    
    fprintf('Accuracy: %.2f%%\n', meanAccuracy * 100);
    
    % Plot general mistakes and generate Grad-CAM for the errors
    wrongFiles = visualize_mistakes(imageStore, lastYTest, lastYPred, lastTestIdx, methodTag);
    if ~isempty(wrongFiles)
        visualize_gradcam_mistakes(wrongFiles, deepNet, methodTag);
    end
end

function [meanAccuracy, lastYTest, lastYPred, lastTestIdx] = execute_cross_validation(features, labels, cvPartition, trainPredictFn)
    accuracies = zeros(cvPartition.NumTestSets, 1);
    for fold = 1:cvPartition.NumTestSets
        trainIdx = cvPartition.training(fold);
        testIdx  = cvPartition.test(fold);
        
        YPred = trainPredictFn(features(trainIdx, :), labels(trainIdx), features(testIdx, :));
        accuracies(fold) = sum(YPred == labels(testIdx)) / length(testIdx);
        
        if fold == cvPartition.NumTestSets
            lastYTest = labels(testIdx);
            lastYPred = YPred;
            lastTestIdx = testIdx;
        end
    end
    meanAccuracy = mean(accuracies);
end

%% ========================================================================
%% Visualization Logic (t-SNE & Grad-CAM)
%% ========================================================================

function visualize_feature_space(features, labels, plotTitle)
    % Compresses features to 2D using t-SNE to prove semantic gap/separation.
    rng('default'); % Ensure reproducibility for the report
    
    try
        % Z-score standardization stabilizes t-SNE across different feature types
        featuresNorm = normalize(features, 'zscore');
        embedded = tsne(featuresNorm, 'Perplexity', min(30, size(featuresNorm,1)-1));
    catch ME
        warning('t-SNE failed for %s: %s', plotTitle, ME.message);
        return;
    end
    
    fig = figure('Name', ['t-SNE: ' plotTitle], 'Visible', 'off');
    gscatter(embedded(:,1), embedded(:,2), labels, 'rbg', 'o*+');
    title(strrep(plotTitle, '_', ' '));
    xlabel('Dim 1'); ylabel('Dim 2');
    grid on;
    
    saveName = sprintf('tsne_%s.png', plotTitle);
    exportgraphics(fig, saveName, 'Resolution', 300);
    close(fig);
    fprintf('    Saved t-SNE: %s\n', saveName);
end



function visualize_gradcam_mistakes(wrongFiles, net, methodTag)
    % Idiosyncrasy: Because the SVM handles the final classification externally, 
    % we ask the network for its *native* top prediction to see what it "thinks" 
    % the image is, and map the attention to that specific concept.
    
    inputSize = net.Layers(1).InputSize;
    maxImagesToAnalyze = min(3, length(wrongFiles)); % Limit output to avoid clutter
    
    for i = 1:maxImagesToAnalyze
        imgFile = wrongFiles{i};
        img = read_safe_image(imgFile);
        imgResized = imresize(img, inputSize(1:2));
        
        % Get the network's own native prediction
        nativeClass = classify(net, imgResized);
        
        try
            % Generate attention map for the predicted class
            scoreMap = gradCAM(net, imgResized, nativeClass);
            
            fig = figure('Name', sprintf('Grad-CAM: %s', methodTag), 'Visible', 'off');
            imshow(imgResized);
            hold on;
            imagesc(scoreMap, 'AlphaData', 0.5);
            colormap('jet');
            hold off;
            
            [~, name, ~] = fileparts(imgFile);
            title(sprintf('Network saw: %s', string(nativeClass)), 'Interpreter', 'none');
            
            saveName = sprintf('gradcam_%s_%s.png', methodTag, name);
            exportgraphics(fig, saveName, 'Resolution', 300);
            close(fig);
            fprintf('    Saved Grad-CAM: %s\n', saveName);
        catch
            warning('Grad-CAM generation failed for %s', imgFile);
        end
    end
end

function wrongFiles = visualize_mistakes(imageStore, YTest, YPred, testIdx, methodTag)
    wrongIdx = find(YPred ~= YTest);
    
    if isempty(wrongIdx)
        wrongFiles = {};
        return;
    end
    
    currentTestFiles = imageStore.Files(testIdx);
    wrongFiles = currentTestFiles(wrongIdx);
    
    thumbnails = cell(1, length(wrongFiles));
    for k = 1:length(wrongFiles)
        thumbnails{k} = read_safe_image(wrongFiles{k});
        thumbnails{k} = imresize(thumbnails{k}, [150 150]); % Uniform size for montage
    end
    
    fig = figure('Name', ['Mistakes ' methodTag], 'Visible', 'off');
    montage(thumbnails, 'ThumbnailSize', [150 150]);
    title(sprintf('%s Mistakes (Total: %d)', methodTag, length(wrongFiles)), 'Interpreter', 'none');
    
    timestamp = datestr(now, 'yyyyMMdd_HHmmss');
    saveName = sprintf('ass1_Mistakes_%s_%s.png', methodTag, timestamp);
    exportgraphics(fig, saveName, 'Resolution', 300);
    close(fig);
end

%% ========================================================================
%% Feature Extraction & Infrastructure
%% ========================================================================

function [balancedStore, minCount] = load_and_balance_data(baseDir, class1, class2)
    dir1 = fullfile(baseDir, class1);
    dir2 = fullfile(baseDir, class2);
    
    if ~exist(dir1, 'dir') || ~exist(dir2, 'dir')
        warning('Directories for %s vs %s not found. Skipping.', class1, class2);
        balancedStore = []; minCount = 0;
        return;
    end
    
    imds1 = imageDatastore(dir1, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imds2 = imageDatastore(dir2, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    minCount = min(length(imds1.Files), length(imds2.Files));
    subStore1 = splitEachLabel(imds1, minCount, 'randomized');
    subStore2 = splitEachLabel(imds2, minCount, 'randomized');
    
    balancedStore = imageDatastore(cat(1, subStore1.Files, subStore2.Files), ...
        'Labels', cat(1, subStore1.Labels, subStore2.Labels));
end

function [net, layer] = initialize_deep_network()
    if ~isempty(which('resnet50'))
        net = resnet50();
        layer = 'avg_pool';
    else
        net = alexnet();
        layer = 'fc7';
    end
end

function features = extract_dcnn_features(imageStore, net, layer)
    inputSize = net.Layers(1).InputSize;
    imageStore.ReadFcn = @(f) read_resized_image(f, inputSize);
    features = activations(net, imageStore, layer, 'OutputAs', 'rows');
end

function [features, metrics] = extract_dense_surf(I)
    features = zeros(1, 64, 'single'); metrics = 0;
    try
        I = ensure_rgb(I);
        grayImage = rgb2gray(I);
        points = generate_random_surf_points(grayImage, 1000);
        [extFeatures, validPoints] = extractFeatures(grayImage, points);
        
        if ~isempty(extFeatures)
            features = extFeatures;
            metrics = validPoints.Metric;
        end
    catch
    end
end

function surfPts = generate_random_surf_points(I, numPoints)
    [sy, sx] = size(I);
    locations = zeros(numPoints, 2);
    scales = zeros(numPoints, 1);
    
    for i = 1:numPoints
        scale = randn() * 3 + 3;
        while scale < 1.6; scale = randn() * 3 + 3; end
        pt = ceil(([sx, sy] - ceil(scale) * 2) .* rand(1, 2) + ceil(scale));
        locations(i, :) = pt;
        scales(i) = scale;
    end
    surfPts = SURFPoints(locations, 'Scale', scales);
end

function histVec = read_hsv_histogram(filename)
    try
        hsv = rgb2hsv(ensure_rgb(imread(filename)));
        idx = floor(hsv(:,:,1)*3.99)*16 + floor(hsv(:,:,2)*3.99)*4 + floor(hsv(:,:,3)*3.99);
        histVec = histcounts(idx, 0:64, 'Normalization', 'pdf');
    catch
        histVec = zeros(1, 64);
    end
end

function I_out = read_resized_image(filename, inputSize)
    try
        I_out = imresize(ensure_rgb(imread(filename)), inputSize(1:2));
    catch
        I_out = zeros([inputSize(1:2), 3], 'uint8');
    end
end

function I_out = read_safe_image(filename)
    try I_out = ensure_rgb(imread(filename)); catch; I_out = zeros(300, 300, 3, 'uint8'); end
end

function I_out = ensure_rgb(I)
    if size(I, 3) == 1; I_out = cat(3, I, I, I); else; I_out = I; end
end

function print_section_header(titleText)
    fprintf('\n%s\n%s\n%s\n', repmat('=', 1, 40), titleText, repmat('=', 1, 40));
end