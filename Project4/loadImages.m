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


%% PCA
[coeff,scoreTrain,~,~,explained,mu] = pca(imagesTrain);
explained

sum_explained = 0;
idx = 0;
while sum_explained < 85
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end
idx
imagesTrain = scoreTrain(:,1:idx);
Mdl = fitcecoc(imagesTrain,labelsTrain,'Coding','onevsall');
imagesTest = bsxfun(@minus,imagesTest,mu)*coeff(:,1:idx);

%%
%Fit model on training data
%Mdl = fitcecoc(imagesTrain,labelsTrain);
%%
%Evaluate training performance
yhatTrain =Mdl.predict(imagesTrain);
classperfTrain = classperf(labelsTrain,yhatTrain);

%Evaluate test performance
yhatTest =Mdl.predict(imagesTest);
classperfTest = classperf(labelsTest,yhatTest);


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
