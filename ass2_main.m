%% Web Image Re-ranking Experiment
% Purpose: Train SVM on Top-N pseudo-positives and re-rank test images with noise injection.
% Principles applied: SOLID (SRP), Readable Code (Low Cognitive Load).

function web_image_reranking_main()
    % main function wrapper to allow local functions
    
    clc; clear; close all;

    %% 1. Configuration
    % Centralized configuration to separate 'Data' from 'Logic'.
    Config = struct();
    Config.Keywords      = {'apple', 'kiwi'};
    Config.Paths.Base    = 'img';
    Config.Paths.Bg      = 'bgimg';
    Config.Counts.Train  = [25, 50];  % Top-N parameters
    Config.Counts.BgTrain= 500;       % Negatives for Training
    Config.Counts.BgTest = 200;       % Negatives for Testing (Noise)

    % Validate Dependencies
    if isempty(which('resnet50')) && isempty(which('alexnet'))
        error('Deep Learning Toolbox model not found. Install ResNet50 or AlexNet.');
    end

    %% 2. Setup Deep Learning Model
    [deepNet, featureLayer, inputSize] = load_pretrained_model();
    fprintf('Model Loaded: %s (Layer: %s)\n', class(deepNet), featureLayer);

    %% 3. Main Experiment Loop
    for k = 1:length(Config.Keywords)
        targetKeyword = Config.Keywords{k};
        print_section_header(sprintf('Processing Keyword: %s', targetKeyword));

        % 3.1 Prepare Background Data (Noise)
        % Using specific splitting logic to avoid data leakage
        [bgTrainFiles, bgTestFiles] = prepare_background_data(Config);
        
        % 3.2 Target Image Discovery
        % Abstraction: We don't care here if it's single or split folders
        [allTrainCandidates, allTestCandidates] = discover_target_images(Config, targetKeyword);
        
        if isempty(allTrainCandidates)
            warning('No images found for %s. Skipping.', targetKeyword);
            continue;
        end

        % 3.3 Iterate through different Top-N definitions
        for nIdx = 1:length(Config.Counts.Train)
            numPosTrain = Config.Counts.Train(nIdx);
            
            % --- Core Logic: Train/Test Split ---
            [posTrainFiles, testFiles] = create_experiment_sets(...
                allTrainCandidates, allTestCandidates, bgTestFiles, numPosTrain);
            
            fprintf('  > Experiment Top-%d: Train[%d Pos, %d Neg] | Test[%d Total]\n', ...
                numPosTrain, length(posTrainFiles), length(bgTrainFiles), length(testFiles));

            % --- Feature Extraction ---
            % Extract features for all sets
            trainPosFeat = extract_features(posTrainFiles, deepNet, featureLayer, inputSize);
            trainNegFeat = extract_features(bgTrainFiles, deepNet, featureLayer, inputSize);
            testFeat     = extract_features(testFiles, deepNet, featureLayer, inputSize);

            if isempty(trainPosFeat) || isempty(trainNegFeat)
                warning('Feature extraction failed (empty set). Skipping Top-%d.', numPosTrain);
                continue;
            end

            % --- SVM Training ---
            svmModel = train_svm_ranker(trainPosFeat, trainNegFeat);

            % --- Prediction & Ranking ---
            [~, scores] = predict(svmModel, testFeat);
            posScores   = scores(:, 2); % Probability of class "1" (Target)

            % --- Save Results ---
            % 1. Original Order (Unsorted)
            save_ranking_results(testFiles, posScores, targetKeyword, numPosTrain, 'original', Config);
            
            % 2. Re-ranked Order (Sorted)
            [sortedScores, sortIdx] = sort(posScores, 'descend');
            sortedFiles = testFiles(sortIdx);
            save_ranking_results(sortedFiles, sortedScores, targetKeyword, numPosTrain, 'reranked', Config);
        end
    end
end

%% ========================================================================
%% Domain Logic Functions (Sub-problems)
%% ========================================================================

function [net, layer, sizeInfo] = load_pretrained_model()
    % Factory method for model loading
    if ~isempty(which('resnet50'))
        net = resnet50();
        layer = 'avg_pool';
    else
        net = alexnet();
        layer = 'fc7';
    end
    sizeInfo = net.Layers(1).InputSize;
end

function [bgTrain, bgTest] = prepare_background_data(Config)
    % Handles loading and splitting of background images safely
    imdsBg = imageDatastore(Config.Paths.Bg);
    allBgFiles = imdsBg.Files(randperm(length(imdsBg.Files))); % Shuffle
    
    nTotal = length(allBgFiles);
    nTrain = min(Config.Counts.BgTrain, floor(nTotal * 0.7));
    nTest  = min(Config.Counts.BgTest, nTotal - nTrain);
    
    bgTrain = allBgFiles(1:nTrain);
    bgTest  = allBgFiles(nTrain+1 : nTrain+nTest);
end

function [trainCandidates, testCandidates] = discover_target_images(Config, keyword)
    % Abstracts the file system structure (Split folders vs Single folder)
    trainDir = fullfile(Config.Paths.Base, [keyword '_train']);
    testDir  = fullfile(Config.Paths.Base, [keyword '_test']);
    
    hasSeparateFolders = exist(trainDir, 'dir') && exist(testDir, 'dir');
    
    if hasSeparateFolders
        imdsTrain = imageDatastore(trainDir);
        imdsTest  = imageDatastore(testDir);
        trainCandidates = sort(imdsTrain.Files);
        testCandidates  = sort(imdsTest.Files);
    else
        % Fallback: Single folder strategy
        targetDir = fullfile(Config.Paths.Base, keyword);
        if ~exist(targetDir, 'dir')
            trainCandidates = {};
            testCandidates = {};
            return;
        end
        imdsFull = imageDatastore(targetDir);
        trainCandidates = sort(imdsFull.Files); % Use all as candidates
        testCandidates  = {}; % Will be handled by splitter
    end
end

function [posTrain, finalTest] = create_experiment_sets(allTrain, allTest, bgTest, numTrain)
    % Logic to define exactly what goes into Training vs Testing
    % Why: To ensure Top-N selection is consistent regardless of folder structure.
    
    % 1. Select Training Positives (Top-N)
    if length(allTrain) <= numTrain
        % Fallback if requested N is larger than available images
        splitIdx = floor(length(allTrain) / 2);
        if isempty(allTest) 
             % Single folder case: Split available into 50/50
             posTrain = allTrain(1:splitIdx);
             remainingForTest = allTrain(splitIdx+1:end);
        else
             % Separate folder case: Use all training
             posTrain = allTrain;
             remainingForTest = {};
        end
    else
        % Standard Top-N case
        posTrain = allTrain(1:numTrain);
        if isempty(allTest)
            % Single folder: The rest become test candidates
            remainingForTest = allTrain(numTrain+1:end);
        else
            % Separate folder: Unused training images are ignored (standard protocol)
            remainingForTest = {};
        end
    end
    
    % 2. Construct Final Test Set (Positives + Noise)
    if isempty(allTest)
        posTest = remainingForTest;
    else
        posTest = allTest;
    end
    
    finalTest = [posTest; bgTest];
end

function svmModel = train_svm_ranker(posFeats, negFeats)
    % Encapsulates SVM training logic
    XTrain = [posFeats; negFeats];
    YTrain = [ones(size(posFeats, 1), 1); -ones(size(negFeats, 1), 1)];
    
    svmModel = fitcsvm(XTrain, YTrain, ...
        'KernelFunction', 'linear', ...
        'Standardize', true);
end

function features = extract_features(fileList, net, layer, inputSize)
    % Wrapper for feature extraction with safe image reading
    if isempty(fileList)
        features = [];
        return;
    end
    
    % Create a temporary datastore with a custom read function
    imds = imageDatastore(fileList);
    imds.ReadFcn = @(f) read_safe_image(f, inputSize);
    
    features = activations(net, imds, layer, 'OutputAs', 'rows');
end

function save_ranking_results(fileList, scores, keyword, numTrain, typeSuffix, Config)
    % Formats and writes the output file
    fileName = sprintf('ass2_rank_%s_%s_top%d.txt', typeSuffix, keyword, numTrain);
    fid = fopen(fileName, 'w');
    
    for i = 1:length(fileList)
        % Create clean relative path for readability in output file
        relPath = get_relative_path(fileList{i}, Config, keyword);
        fprintf(fid, '%s %.4f\n', relPath, scores(i));
    end
    
    fclose(fid);
    fprintf('    Saved: %s\n', fileName);
end

%% ========================================================================
%% Low-Level Helper Functions (Infrastructure)
%% ========================================================================

function relPath = get_relative_path(absPath, Config, keyword)
    % Determines how to display the file path based on its origin
    [~, fname, ext] = fileparts(absPath);
    
    if contains(absPath, Config.Paths.Bg)
        relPath = fullfile(Config.Paths.Bg, [fname ext]);
    elseif contains(absPath, [keyword '_test'])
        relPath = fullfile([keyword '_test'], [fname ext]);
    elseif contains(absPath, [keyword '_train'])
        relPath = fullfile([keyword '_train'], [fname ext]);
    else
        relPath = fullfile(Config.Paths.Base, keyword, [fname ext]);
    end
end

function I_out = read_safe_image(filename, targetSize)
    % Robust image reader: Handles corruption, Grayscale->RGB, and Resizing
    try
        I = imread(filename);
        
        % Force 3 channels (RGB)
        if size(I, 3) == 1
            I = cat(3, I, I, I);
        end
        
        I_out = imresize(I, targetSize(1:2));
        
    catch
        [~, name, ext] = fileparts(filename);
        warning('Corrupt image skipped: %s%s', name, ext);
        I_out = zeros([targetSize(1:2), 3], 'uint8'); % Return black image
    end
end

function print_section_header(titleText)
    fprintf('\n%s\n%s\n%s\n', repmat('=',1,40), titleText, repmat('=',1,40));
end