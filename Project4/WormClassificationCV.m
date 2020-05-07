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
   [x1,y1,z1] = size(currentimage);
   if (abs(x1-y1) <1 )
       currentimageResized = imresize(currentimage,[imageSize imageSize]);
       ImgVector = currentimageResized(:);
       wormimages = [wormimages; ImgVector'];
       wormlabels = [wormlabels; 1];
   end
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
   [x1,y1,z1] = size(currentimage);
   if (abs(x1-y1) <1 )  
       currentimageResized = imresize(currentimage,[imageSize imageSize]);
       ImgVector = currentimageResized(:);
       nowormimages = [nowormimages; ImgVector'];
       nowormlabels = [nowormlabels; 0];
   end
end

%% Merge the datasets and shuffle the data
%Merge
fullImageSet = [wormimages; nowormimages];
fullLabelSet = [wormlabels; nowormlabels];

% Set random seed so same split is used
rng(10);

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
c = cvpartition(fullLabelSetShuffled,'KFold',5);


sumTrain = 0;
sumTest = 0;
sumTrainPCA = 0;
sumTestPCA = 0;
sumTestSen =0;
sumTestSpec =0;
sumTestSenPCA =0;
sumTestSpecPCA =0;
sumTrainSen =0;
sumTrainSpec =0;
sumTrainSenPCA =0;
sumTrainSpecPCA =0;
for i = 1:c.NumTestSets
    idx = c.test(i);
    %Obtain train and test sets using index
    % Preprocess images: convert to double, normalize if it helps
    XTrain = im2double(fullImageSetShuffled(~idx,:));
    XTest  = im2double(fullImageSetShuffled(idx,:));
    % XTrain = normalize(XTrain,'zscore');
    % XTest  = normalize(XTest,'zscore');
    yTrain = fullLabelSetShuffled(~idx,:);
    yTest  = fullLabelSetShuffled(idx,:);
%% Fit SVM classifier without PCA and evaluate train and test performance
%     Mdl = fitcsvm(XTrain,yTrain,'KernelFunction','gaussian','Standardize',false,'OptimizeHyperparameters',{'BoxConstraint','KernelScale'},'HyperparameterOptimizationOptions',struct('UseParallel',true));
    tic
    Mdl = fitcsvm(XTrain,yTrain,'KernelFunction','gaussian','Standardize',false,'BoxConstraint',7.3,'KernelScale',2.8);
    tFit(i) = toc;
    % 
    % Evaluate train performance
    yHatTrain = Mdl.predict(XTrain);
    classperfTrain = classperf(yTrain,yHatTrain);
    sumTrain =sumTrain+ classperfTrain.CorrectRate;
    sumTrainSen =sumTrainSen+ classperfTrain.Sensitivity;
    sumTrainSpec =sumTrainSpec+ classperfTrain.Specificity;
    % Evaluate test performance
    tic
    yHatTest = Mdl.predict(XTest);
    tTest(i) = toc;
    classperfTest = classperf(yTest,yHatTest);
    sumTest =sumTest+ classperfTest.CorrectRate;
    sumTestSen =sumTestSen+ classperfTest.Sensitivity;
    sumTestSpec =sumTestSpec+ classperfTest.Specificity;
%% Fit SVM classifier with PCA and evaluate train and test performance

    %Train PCA
    [coeff,scoreTrain,~,~,explained,mu] = pca(XTrain);

    %Find # of components to equal X% variance of original data
    sum_explained = 0;
    idx = 0;
    while sum_explained < 87.1
        idx = idx + 1;
        sum_explained = sum_explained + explained(idx);
    end
    idx

    % Project onto new PCA basis/components
    XTrainPCA = scoreTrain(:,1:idx);
    XTestPCA = bsxfun(@minus,XTest,mu)*coeff(:,1:idx);

    %Fit SVC
    % MdlPCA = fitcsvm(XTrainPCA,yTrain,'KernelFunction','gaussian','Standardize',false,'OptimizeHyperparameters',{'BoxConstraint','KernelScale'},'HyperparameterOptimizationOptions',struct('UseParallel',true));
    tic
    MdlPCA = fitcsvm(XTrainPCA,yTrain,'KernelFunction','gaussian','Standardize',false,'BoxConstraint',2,'KernelScale',2);
    tFitPCA(i) = toc;
    % Evaluate train performance
    yHatTrainPCA = MdlPCA.predict(XTrainPCA);
    classperfTrainPCA = classperf(yTrain,yHatTrainPCA);
    sumTrainPCA =sumTrainPCA+ classperfTrainPCA.CorrectRate;
    sumTrainSenPCA =sumTrainSenPCA+ classperfTrainPCA.Sensitivity;
    sumTrainSpecPCA =sumTrainSpecPCA+ classperfTrainPCA.Specificity;
    % Evaluate test performance
    tic
    yHatTestPCA = MdlPCA.predict(XTestPCA);
    tTestPCA(i) = toc;
    classperfTestPCA = classperf(yTest,yHatTestPCA);
    sumTestPCA =sumTestPCA+ classperfTestPCA.CorrectRate;
    sumTestSenPCA =sumTestSenPCA+ classperfTestPCA.Sensitivity;
    sumTestSpecPCA =sumTestSpecPCA+ classperfTestPCA.Specificity;
end
%%
CVTrainScore = sumTrain/c.NumTestSets;
CVTestScore = sumTest/c.NumTestSets;
CVTrainPCAScore =sumTrainPCA/c.NumTestSets;
CVTestPCAScore = sumTestPCA/c.NumTestSets;

CVTrainSen = sumTrainSen/c.NumTestSets;
CVTestSen = sumTestSen/c.NumTestSets;
CVTrainPCASen =sumTrainSenPCA/c.NumTestSets;
CVTestPCASen = sumTestSenPCA/c.NumTestSets;

CVTrainSpec = sumTrainSpec/c.NumTestSets;
CVTestSpec = sumTestSpec/c.NumTestSets;
CVTrainPCASpec =sumTrainSpecPCA/c.NumTestSets;
CVTestPCASSpec = sumTestSpecPCA/c.NumTestSets;


ttestAvg = sum(tTest)/5;
ttrainAvg = sum(tFit)/5;
ttestPCAAvg = sum(tTestPCA)/5;
ttrainPCAAvg = sum(tFitPCA)/5;


