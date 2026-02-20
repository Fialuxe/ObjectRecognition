% ass2_visualize_svm_score.m
% Task 2: Visualize SVM Confidence and Rank Sensitivity

% 1. Setup
keyword = 'apple';
N = 50; % Top-N Training
net = resnet50;
layer = 'avg_pool';

% 2. Prepare Data (Training)
disp('Preparing Training Data...');
pos_files = dir(fullfile('img', [keyword '_train'], '*.jpg'));
neg_files = dir(fullfile('bgimg', '*.jpg'));

% Select Top-N
train_pos_files = pos_files(1:N);
train_neg_files = neg_files(randperm(length(neg_files), 500)); % Random 500 Neg

% Extract Features
train_features = [];
train_labels = [];

for i = 1:length(train_pos_files)
    img = imread(fullfile(train_pos_files(i).folder, train_pos_files(i).name));
    img = imresize(img, net.Layers(1).InputSize(1:2));
    feat = activations(net, img, layer, 'OutputAs', 'rows');
    train_features = [train_features; feat];
    train_labels = [train_labels; 1];
end

for i = 1:length(train_neg_files)
    img = imread(fullfile(train_neg_files(i).folder, train_neg_files(i).name));
    img = imresize(img, net.Layers(1).InputSize(1:2));
    feat = activations(net, img, layer, 'OutputAs', 'rows');
    train_features = [train_features; feat];
    train_labels = [train_labels; -1];
end

% Train SVM
disp('Training SVM...');
model = fitcsvm(train_features, train_labels, 'KernelFunction', 'linear', 'Standardize', true);

% 3. Evaluate Test Data
disp('Evaluating Test Data...');
test_pos_files = dir(fullfile('img', [keyword '_test'], '*.jpg'));
test_neg_files = neg_files(randperm(length(neg_files), 200)); % Random 200 Neg (different set ideally)

test_features = [];
test_labels_gt = [];
test_paths = {};

for i = 1:length(test_pos_files)
    path = fullfile(test_pos_files(i).folder, test_pos_files(i).name);
    try
        img = imread(path);
        img = imresize(img, net.Layers(1).InputSize(1:2));
        feat = activations(net, img, layer, 'OutputAs', 'rows');
        test_features = [test_features; feat];
        test_labels_gt = [test_labels_gt; 1];
        test_paths{end+1} = path;
    catch
        warning(['Failed to read ', path]);
    end
end

for i = 1:length(test_neg_files)
    path = fullfile(test_neg_files(i).folder, test_neg_files(i).name);
    try
        img = imread(path);
        img = imresize(img, net.Layers(1).InputSize(1:2));
        feat = activations(net, img, layer, 'OutputAs', 'rows');
        test_features = [test_features; feat];
        test_labels_gt = [test_labels_gt; -1];
        test_paths{end+1} = path;
    catch
        warning(['Failed to read ', path]);
    end
end

% Predict Scores
[~, score] = predict(model, test_features);
scores = score(:, 2); % Positive Class Score

% 4. Visualization
figure('Position', [100, 100, 800, 600]);

% Histogram of Scores
subplot(2, 1, 1);
histogram(scores(test_labels_gt == 1), 20, 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
histogram(scores(test_labels_gt == -1), 20, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title(['SVM Score Distribution (Top-' num2str(N) ' Training)']);
xlabel('Score (Distance to Boundary)'); ylabel('Count');
legend('Positive (GT)', 'Negative (GT)');
grid on;

% Sort by Score
[sorted_scores, idx] = sort(scores, 'descend');
sorted_labels = test_labels_gt(idx);
sorted_paths = test_paths(idx);

% Display Top-10 Ranked Images (Visual Check)
subplot(2, 1, 2);
axis off;
title('Top-10 Ranked Images');
for k = 1:10
    axes('Position', [(k-1)*0.1, 0.05, 0.09, 0.2]);
    imshow(imread(sorted_paths{k}));
    if sorted_labels(k) == 1
        title(['Score: ' num2str(sorted_scores(k), '%.2f')], 'Color', 'blue');
    else
        title(['Score: ' num2str(sorted_scores(k), '%.2f')], 'Color', 'red');
    end
    axis off;
end

% Save Figure
saveas(gcf, 'ass2_svm_score_dist.png');
disp('SVM Score visualization saved to ass2_svm_score_dist.png');
