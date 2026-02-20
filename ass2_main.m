%% Web Image Re-ranking Experiment
% Purpose: Train linear SVM on top-N positive images and background negatives, 
%          then re-rank noisy test images.

function web_image_reranking_main()
    clc; clear; close all;

    %% 1. Configuration
    % Centralized configuration to separate 'Data' from 'Logic'.
    Config = struct();
    Config.Keywords      = {'apple', 'kiwi'};
    Config.Paths.Base    = 'img';
    Config.Paths.Bg      = 'bgimg';   % Background images in current directory
    Config.Counts.Train  = [25, 50];  % Top-N parameters for positive samples
    Config.Counts.BgTrain= 500;       % Negatives for Training

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

        % 3.1 Prepare Datasets
        bgTrainFiles = prepare_background_data(Config);
        [allTrainCandidates, testFiles] = discover_target_images(Config, targetKeyword);
        
        if isempty(allTrainCandidates) || isempty(testFiles)
            warning('Images not found for %s. Skipping.', targetKeyword);
            continue;
        end

        % 3.2 Iterate through different Top-N definitions
        for nIdx = 1:length(Config.Counts.Train)
            numPosTrain = Config.Counts.Train(nIdx);
            
            % Select top N images from Bing data (Train)
            posTrainFiles = select_top_n(allTrainCandidates, numPosTrain);
            
            fprintf('  > Experiment Top-%d: Train[%d Pos, %d Neg] | Test[%d Total]\n', ...
                numPosTrain, length(posTrainFiles), length(bgTrainFiles), length(testFiles));

            % 3.3 Feature Extraction
            trainPosFeat = extract_features(posTrainFiles, deepNet, featureLayer, inputSize);
            trainNegFeat = extract_features(bgTrainFiles, deepNet, featureLayer, inputSize);
            testFeat     = extract_features(testFiles, deepNet, featureLayer, inputSize);

            if isempty(trainPosFeat) || isempty(trainNegFeat)
                warning('Feature extraction failed. Skipping Top-%d.', numPosTrain);
                continue;
            end

            % 3.4 SVM Training
            svmModel = train_svm_ranker(trainPosFeat, trainNegFeat);

            % 3.5 Prediction & Ranking on Flickr data (Test)
            [~, scores] = predict(svmModel, testFeat);
            posScores   = scores(:, 2); % Probability of target class

            % 3.6 Save Results
            save_ranking_results(testFiles, posScores, targetKeyword, numPosTrain, 'original', Config);
            
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

function bgTrain = prepare_background_data(Config)
    % Extracts a fixed number of random background images for negative samples.
    imdsBg = imageDatastore(Config.Paths.Bg);
    shuffledFiles = imdsBg.Files(randperm(length(imdsBg.Files))); 
    
    nTrain = min(Config.Counts.BgTrain, length(shuffledFiles));
    bgTrain = shuffledFiles(1:nTrain);
end

function [trainFiles, testFiles] = discover_target_images(Config, keyword)
    % Loads files directly from explicit _train and _test directories
    trainDir = fullfile(Config.Paths.Base, [keyword '_train']);
    testDir  = fullfile(Config.Paths.Base, [keyword '_test']);
    
    if exist(trainDir, 'dir') && exist(testDir, 'dir')
        imdsTrain = imageDatastore(trainDir);
        imdsTest  = imageDatastore(testDir);
        trainFiles = sort(imdsTrain.Files);
        testFiles  = sort(imdsTest.Files);
    else
        trainFiles = {};
        testFiles = {};
    end
end

function topNFiles = select_top_n(allFiles, n)
    % Safely slice the top N files without exceeding array bounds
    actualN = min(n, length(allFiles));
    topNFiles = allFiles(1:actualN);
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
    if isempty(fileList)
        features = [];
        return;
    end
    
    % Create a temporary datastore with a custom read function for robustness
    imds = imageDatastore(fileList);
    imds.ReadFcn = @(f) read_safe_image(f, inputSize);
    
    features = activations(net, imds, layer, 'OutputAs', 'rows');
end

function save_ranking_results(fileList, scores, keyword, numTrain, typeSuffix, Config)
    fileName = sprintf('ass2_rank_%s_%s_top%d.txt', typeSuffix, keyword, numTrain);
    fid = fopen(fileName, 'w');
    
    for i = 1:length(fileList)
        relPath = extract_relative_path(fileList{i}, Config.Paths.Base);
        fprintf(fid, '%s %f\n', relPath, scores(i));
    end
    
    fclose(fid);
    fprintf('    Saved: %s\n', fileName);
end

%% ========================================================================
%% Low-Level Helper Functions (Infrastructure)
%% ========================================================================

function relPath = extract_relative_path(absPath, baseFolder)
    % Formats path for clean text output (e.g., 'img/apple_test/000001.jpg')
    idx = strfind(absPath, [filesep baseFolder filesep]);
    if ~isempty(idx)
        relPath = absPath(idx(end)+1:end); 
    else
        % Fallback for bgimg folder
        [parentDir, fname, ext] = fileparts(absPath);
        [~, parentName] = fileparts(parentDir);
        relPath = fullfile(parentName, [fname ext]);
    end
    % Ensure consistent forward slashes in text output
    relPath = strrep(relPath, '\', '/');
end

function I_out = read_safe_image(filename, targetSize)
    % Robust image reader: Handles corruption, Grayscale->RGB, and Resizing
    try
        I = imread(filename);
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