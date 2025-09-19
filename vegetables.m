%Begin
veggiesDatasetPath = fullfile(matlabroot,'toolbox','nnet','vegetables');
imds = imageDatastore(veggiesDatasetPath, ...
'IncludeSubfolders',true,'LabelSource','foldernames');
%Show random 20 images
figure;
perm = randperm(400,20);
for i = 1:20
subplot(4,5,i);
imshow(imds.Files{perm(i)});
end

% Define the input and output directories 
inputDir = 'C:\Users\nurin\Desktop\vegetables\tomato';  % Replace with your input directory 
outputDir = 'C:\Users\nurin\Desktop\vegetables\tomato\New folder';  % Replace with your output directory 
 
% Ensure the output directory exists 
if ~exist(outputDir, 'dir') 
    mkdir(outputDir); 
end 
 
% List all .jpg files in the directory 
imagefiles = dir(fullfile(inputDir, '*.jpg')); 
 
% Loop through each .jpg file and resize it 
for i = 1 : length(imagefiles) 
    % Read the image 
    filename = fullfile(inputDir, imagefiles(i).name); 
    sf = imread(filename); 
     
    % Resize the image to a standard size, e.g., 100x100 
    sf = imresize(sf, [100 100]); 
     
    % Create a new file name 
    newName = sprintf('%04d.jpg', i);  % Use '%04d' for zero-padded numbering 
    newFilename = fullfile(outputDir, newName); 
     
    % Save the resized image to the output directory 
    imwrite(sf, newFilename); 
     
    % Display a message to track progress 
    fprintf('Processed image %d: %s\n', i, imagefiles(i).name); 
end

%%Count total image of each type of vegetables
labelCount = countEachLabel(imds)

img = readimage(imds,1);
size(img)

numTrainFiles = 75; 
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize'); 

layers = [ 
imageInputLayer([100 100 3]) 
 
  convolution2dLayer(25,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(13,'Stride',2)
    
    convolution2dLayer(25,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(13,'Stride',2)
    
    convolution2dLayer(23,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer
]

% Architecture 1
options1 = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
net1 = trainNetwork(imdsTrain, layers, options1);
%% 
load('trainedNet.mat');
inputSize = net1.Layers(1).InputSize

% Provide the full path to the image 
imagePath = 'C:\Users\nurin\Downloads\Sem III\CSC583\vegetables\carrot.jpg';  % Reserved image
 
% Read the image 
I = imread(imagePath); 
figure 
imshow(I)

% Display the size of the image 
disp(size(I)) 

% Resize the image to match the network's input size 
I = imresize(I, inputSize(1:2)); 
%% 
% Classify the image 
[label,scores] = classify(net1, I); 
disp(label) 

% Display the image with the predicted label and confidence score 
figure 
imshow(I) 
classNames = net1.Layers(end).ClassNames; 
title(string(label) + ", " + num2str(100 * scores(classNames == label), 3) + "%"); 

% Display a bar graph with the top 5 predictions 
[~, idx] = sort(scores, 'descend'); 
idx = idx(4:-1:1); 
classNamesTop = net1.Layers(end).ClassNames(idx); 
scoresTop = scores(idx); 
figure 
barh(scoresTop) 
xlim([0 1]) 
title('Top 4 Predictions') 
xlabel('Probability') 
yticklabels(classNamesTop)