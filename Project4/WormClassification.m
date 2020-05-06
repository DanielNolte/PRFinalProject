close all
clear all


%% Load worm images and create labels
pathname = 'data/worm/'; %Update filenames to run locally
imagefiles = dir(fullfile(pathname,'*.jpg'));      
nfiles = length(imagefiles);    % Number of files found
wormimages = [];
wormlabels = [];
imageSize = 30;
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(fullfile(pathname,currentfilename));
   currentimageResized = imresize(currentimage,[imageSize imageSize]);
   ImgVector = currentimageResized(:);
   wormimages = [wormimages; ImgVector'];
   wormlabels = [wormlabels; 1];
end

%% Load noworm images and create labels
pathname1 = 'data/noworm/';
imagefiles = dir(fullfile(pathname1,'*.jpg'));      
nfiles = length(imagefiles);    % Number of files found
nowormimages = [];
nowormlabels = [];
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(fullfile(pathname1,currentfilename));
   currentimageResized = imresize(currentimage,[imageSize imageSize]);
   ImgVector = currentimageResized(:);
   nowormimages = [nowormimages; ImgVector'];
   nowormlabels = [nowormlabels; 0];
end

%% Merge the datasets and shuffle the data
%Merge
fullImageSet = [wormimages; nowormimages];
fullLabelSet = [wormlabels; nowormlabels];

%Set random seed so same train/test split is used
rng(0);

%Shuffle the data and labels
rand_pos = randperm(length(fullLabelSet));
fullLabelSetShuffled = [];
fullImageSetShuffled = [];
for k = 1:length(fullLabelSet)
    pos = rand_pos(k);
    fullImageSetShuffled = [fullImageSetShuffled; fullImageSet(pos,:)];
    fullLabelSetShuffled = [fullLabelSetShuffled; fullLabelSet(pos)];
end
%% Partition data into train and test sets
% Create cv spliter
rng(0);
c = cvpartition(fullLabelSetShuffled,'HoldOut',0.2);
idx = c.test;

%Obtain train and test sets using index
% Preprocess images: convert to double, normalize if it helps
XTrain = im2double(fullImageSetShuffled(~idx,:));
XTest  = im2double(fullImageSetShuffled(idx,:));
% XTrain = normalize(XTrain,'zscore');
% XTest  = normalize(XTest,'zscore');
yTrain = fullLabelSetShuffled(~idx,:);
yTest  = fullLabelSetShuffled(idx,:);
%% Fit SVM classifier without PCA and evaluate train and test performance
% Mdl = fitcsvm(XTrain,yTrain,'KernelFunction','gaussian','Standardize',false,'OptimizeHyperparameters',{'BoxConstraint','KernelScale'},'HyperparameterOptimizationOptions',struct('UseParallel',true));
Mdl = fitcsvm(XTrain,yTrain,'KernelFunction','gaussian','Standardize',false,'BoxConstraint',8.3,'KernelScale',2.8);
% 
% Evaluate train performance
yHatTrain = Mdl.predict(XTrain);
classperfTrain = classperf(yTrain,yHatTrain);
% Evaluate test performance
yHatTest = Mdl.predict(XTest);
classperfTest = classperf(yTest,yHatTest);

%% Fit SVM classifier with PCA and evaluate train and test performance

%Train PCA
[coeff,scoreTrain,~,~,explained,mu] = pca(XTrain);

%Find # of components to equal X% variance of original data
sum_explained = 0;
idx = 0;
while sum_explained < 80
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end
idx

% Project onto new PCA basis/components
XTrainPCA = scoreTrain(:,1:idx);
XTestPCA = bsxfun(@minus,XTest,mu)*coeff(:,1:idx);

%Fit SVC
% MdlPCA = fitcsvm(XTrainPCA,yTrain,'KernelFunction','gaussian','Standardize',false,'OptimizeHyperparameters',{'BoxConstraint','KernelScale'},'HyperparameterOptimizationOptions',struct('UseParallel',true));
MdlPCA = fitcsvm(XTrainPCA,yTrain,'KernelFunction','gaussian','Standardize',false,'BoxConstraint',5,'KernelScale',2);
% Evaluate train performance
yHatTrainPCA = MdlPCA.predict(XTrainPCA);
classperfTrainPCA = classperf(yTrain,yHatTrainPCA);
% Evaluate test performance
yHatTestPCA = MdlPCA.predict(XTestPCA);
classperfTestPCA = classperf(yTest,yHatTestPCA);

