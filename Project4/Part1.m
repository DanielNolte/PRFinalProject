close all
clear all
%Load training data
filename = 'MNISTDigits/train-images-idx3-ubyte/train-images.idx3-ubyte'; %Update filenames to run locally
imagesTrain = loadMNISTImages(filename);
filename1 = 'MNISTDigits/train-labels-idx1-ubyte/train-labels.idx1-ubyte';
labelsTrain = loadMNISTLabels(filename1);

%Load test data
filename = 'MNISTDigits/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte';
imagesTest = loadMNISTImages(filename);
filename1 = 'MNISTDigits/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte';
labelsTest = loadMNISTLabels(filename1);

%Clip training data to first 30,000 samples
% imagesTrain = imagesTrain(1:30000,:);
% labelsTrain = labelsTrain(1:30000,:);
%% PCA

%Train PCA
[coeff,scoreTrain,~,~,explained,mu] = pca(imagesTrain);

%Find # of components to equal 82.5% variance of original data
sum_explained = 0;
idx = 0;
while sum_explained < 82.5
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end
idx

%Transform training data to new PCA basis/components
imagesTrain = scoreTrain(:,1:idx);

%Hyperparameter selection
% t = templateSVM('KernelFunction','gaussian');
% Mdl = fitcecoc(imagesTrain,labelsTrain,'Learners',t,'Options',statset('UseParallel',true),'OptimizeHyperparameters',{'BoxConstraint','KernelScale'},'HyperparameterOptimizationOptions',struct('UseParallel',true));% ,'Coding','onevsall');

%Train model with best parameters
t = templateSVM('KernelFunction','gaussian','BoxConstraint',7.39,'KernelScale',6.7);
tic
Mdl = fitcecoc(imagesTrain,labelsTrain,'Learners',t,'Options',statset('UseParallel',true),'Verbose',1);
TrainTime = toc;
% Transform test data to new PCA components
imagesTest = bsxfun(@minus,imagesTest,mu)*coeff(:,1:idx);

%%
%Evaluate training performance

yhatTrain =Mdl.predict(imagesTrain);
classperfTrain = classperf(labelsTrain,yhatTrain);
trainScore = classperfTrain.CorrectRate;
trainSen = classperfTrain.Sensitivity;
trainSpec = classperfTrain.Specificity;

%Evaluate test performance
tic
yhatTest =Mdl.predict(imagesTest);
TestTime = toc;
classperfTest = classperf(labelsTest,yhatTest);
testScore = classperfTest.CorrectRate;
testSen = classperfTest.Sensitivity;
testSpec = classperfTest.Specificity;

%%
function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images
    fp = fopen(filename,'rb');
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);
    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);
    fclose(fp);
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    % Convert to double and rescale to [0,1]
    images = double(images') / 255;
end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count');
fclose(fp);
end
