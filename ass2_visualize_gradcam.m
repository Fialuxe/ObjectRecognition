function ass2_visualize_gradcam()
    rankFilePath = 'ass2_rank_reranked_apple_top50.txt';
    networkModel = resnet50;
    topNToVisualize = 6;
    
    [imagePaths, imageScores] = readRankData(rankFilePath);
    visualizeGradCam(imagePaths, imageScores, networkModel, topNToVisualize);
end

function [imagePaths, imageScores] = readRankData(filePath)
    fileId = fopen(filePath, 'r');
    if fileId == -1
        error('Could not open rank file: %s', filePath);
    end
    
    % textscan is used for fast parsing of space-delimited text files
    rankData = textscan(fileId, '%s %f', 'Delimiter', ' ');
    fclose(fileId);
    
    imagePaths = rankData{1};
    imageScores = rankData{2};
end

function visualizeGradCam(imagePaths, imageScores, networkModel, topN)
    figure('Position', [100, 100, 1500, 600]);
    inputSize = networkModel.Layers(1).InputSize(1:2);
    numImages = min(topN, length(imagePaths));
    
    for rankIndex = 1:numImages
        imagePath = imagePaths{rankIndex};
        score = imageScores(rankIndex);
        
        if ~exist(imagePath, 'file')
            warning('File not found: %s. Skipping.', imagePath);
            continue;
        end
        
        rawImage = imread(imagePath);
        
        % ResNet-50 strictly requires 3-channel RGB. 
        % Replicate single channel across 3 dimensions to prevent classify/gradCAM crash on grayscale images.
        if size(rawImage, 3) == 1
            rawImage = repmat(rawImage, [1, 1, 3]);
        end
        
        resizedImage = imresize(rawImage, inputSize);
        
        % classify outputs [CategoricalPrediction, Probabilities]. 
        [predictedClass, ~] = classify(networkModel, resizedImage);
        
        % gradCAM accepts the categorical class label directly to compute gradients
        scoreMap = gradCAM(networkModel, resizedImage, predictedClass);
        
        renderSubplot(rankIndex, resizedImage, scoreMap, score, predictedClass);
    end
    
    saveas(gcf, 'ass2_gradcam_top_ranked.png');
    disp('Grad-CAM visualization saved to ass2_gradcam_top_ranked.png');
end

function renderSubplot(plotIndex, imageObject, heatmap, score, predictedClass)
    % Overlays the gradCAM heatmap onto the original image
    subplot(2, 3, plotIndex);
    imshow(imageObject);
    hold on;
    imagesc(heatmap, 'AlphaData', 0.5);
    colormap jet;
    title({sprintf('Rank: %d Score: %.2f', plotIndex, score), ...
           sprintf('ResNet Pred: %s', char(predictedClass))}, 'Interpreter', 'none');
    hold off;
end