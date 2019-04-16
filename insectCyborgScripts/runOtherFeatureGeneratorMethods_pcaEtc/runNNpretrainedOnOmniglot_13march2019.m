% Apply NN, but first pre-training on omniglot to set weights:
% Order of events:
% 0. Select training and test data
% 1. Train NN on Omniglot. Combine the omniglot into metaclasses to have
% lots of training data
% 2. Using these initial weights, Train and test NN on the training and
%     val sets  (for baseline, use ordinary weight initializations)

% Dependencies: Matlab, Statistics and machine learning toolbox, Signal processing toolbox
% Copyright (c) 2019 Charles B. Delahunt
% MIT License

close all
clear

counter = 0;

%% USER ENTRIES:

resultsFolder = 'comparisonFeatureGeneratorsResultsFolder';

% DANGER!!!!! USING THE SAME RESULTSFILENAME WILL OVERWRITE PREVIOUS RESULTS!
resultsFilename = 'nnPreTrainedOnOmni_1';  % results for all cyborgs will be saved in one .mat file  

numTrainList = [ 1 2 3 5 7 10 15 20 30 40 50 70 100 ];  % loop through these
numRuns = 13;

% END OF USER ENTRIES

%----------------------------------------------------------------------------------------------------------

% Some other parameters that do not need to be tweaked to run the code: 


% We only run NN (not svm or nearest neighbors) for this method:
runNeuralNet = 1;

nTe = 21;        % numTest. Test = Holdout (used interchangeably)
nF  = 85;               % number of pixel features
 
showThumbnailsUsed = 0;
nC = 10;  % number of classes
classLabels = 1:nC; 
 
% Neural net parameters:
numHiddenUnits = 85; % placeholder, changes after we see if ENs are being used or not.  % assume one hidden layer
trainRatio = 1.0;   % use all training samples for training (since we have only a few)

%% Generate Omniglot dataset:
% Parameters:
% Parameters required for the dataset generation function are attached to a struct preP.
% 1. The images used. This includes pools for mean-subtraction, baseline, train, and val. 
%     This is NOT the number of training samples per class. That is nTr, defined above. 

% specify pools of indices from which to draw baseline, train, val sets.
 
indPoolForBaseline = 1:5 ;
indPoolForTrain = 1:20; 
indPoolForPostTrain =  1:20;     % this gets redefined as the complement of the training samples, around line 197.

% Population preprocessing pools of indices:
preP.indsToAverageGeneral = [];
preP.indsToCalculateReceptiveField = 1:20;
preP.maxInd = max( [ preP.indsToCalculateReceptiveField,  indPoolForTrain ] );   % we'll throw out unused samples.

% 2. Pre-processing parameters for the thumbnails:
% 
preP.downsampleRate = 5;
preP.crop = 15;
preP.numFeatures =  nF;  % number of pixels in the receptive field
 
preP.pixelSum = 6;  % for use with 'generateDownsampledMnistSet_fn'
preP.targetedPercentile = 90;  % for use with 'generateDownsampledMnistSet_normalizeByMaxVal_fn'
preP.targetedPercentileValue = 0.5; % for use with 'generateDownsampledMnistSet_normalizeByMaxVal_fn'

preP.smearPreDownsample = 0; %    1;
preP.preDownsampleSmearGaussianSigma = 0;   % 4;
preP.smearPostDownsample = 0;
preP.postDownsampleSmearGaussianSigma = 0;  % 0.5;

preP.showAverageImages = 0; 
preP.downsampleMethod = 1; %  0; % 0 means sum square patches of pixels. 1 means use bicubic interpolation.

preP.classLabels = classLabels; % append
preP.useExistingConnectionMatrices = 0; % append

% generate the data array:
 [ fA_full, activePixelInds, lengthOfSide ] = generateDownsampledOmniglotSet_normalizeByMaxVal_fn(preP); % argin = preprocessingParams
 % The dataset fA_full is a feature array ready for running experiments. Each experiment uses a random draw from this dataset.
% fA_full = n x m x numClasses array where n = #active pixels, m = #digits, numClasses = 136.  
  
  
  
%% Generate the dataset from MNIST:
% Parameters:
% Parameters required for the dataset generation function are attached to a struct preP.
% 1. The images used. This includes pools for mean-subtraction, baseline, train, and val. 
%     This is NOT the number of training samples per class. That is nTr, defined above. 

% specify pools of indices from which to draw baseline, train, val sets.
 
indPoolForTrain = 1 :300;  
indPoolForHoldout = 301:550;

% Population preprocessing pools of indices. Note these are used to create the global dataset, ie they are not part of training:
preP.indsToAverageGeneral = 551:1000;
preP.indsToCalculateReceptiveField = 551:1000;
preP.maxInd = max( [ preP.indsToCalculateReceptiveField,  indPoolForTrain ] );   % we'll throw out unused samples.

% 2. Pre-processing parameters for the thumbnails:
preP.downsampleRate = 2;
preP.crop = 2;
preP.numFeatures =  nF;  % number of pixels in the receptive field
preP.pixelSum = 6;
preP.showAverageImages = 0; % showAverageImages; 
preP.downsampleMethod = 1; % 0 means sum square patches of pixels. 1 means use bicubic interpolation.

preP.classLabels = classLabels; % append 

% for back-compatibility:
preP.useExistingConnectionMatrices = 0;

% generate the data array:
 [ fA, activePixelInds, lengthOfSide ] = generateDownsampledMnistSet_fn(preP); % argin = preprocessingParams
% The dataset fA is a feature array ready for running experiments. Each experiment uses a random draw from this dataset.
% fA = n x m x 10 array where n = #active pixels, m = #digits 
%   from each class that will be used. The 3rd dimension gives the class, 1:10   where 10 = '0'.

%% Loop through the number of simulations specified:
for nTr = numTrainList
for ind = 1:numRuns  
    
	if ind == 1
    	disp( [ 'starting NN with omni preTrain experiment, nTr = ', num2str(nTr) ,':' ] )
	end

	counter = counter + 1;
	%% Subsample the dataset for this simulation:

	
   
    % 1. Training images:
    % choose some images from the trainingIndPool:
    rangeTopEnd = max(indPoolForTrain) - min(indPoolForTrain) + 1; 
    trainInds = min(indPoolForTrain) + randsample( rangeTopEnd, nTr ) - 1; 
    % 2.  Test images  
     % choose some images from the postTrainIndPool:
    rangeTopEnd = max(indPoolForHoldout) - min(indPoolForHoldout) + 1;  
    holdoutInds = min(indPoolForHoldout) + randsample( rangeTopEnd, nTe ) - 1; 
 
    % save the actual pixel values of images used: 
    trainIms = fA( :, trainInds, : ); 
    testIms = fA( :, holdoutInds, : );
    % these 3-d arrays get rearranged into 2-d arrays below.
    
    
    %% set up matrices for ML:
    
    % Train:
    
    % X matrices with pixel features only:
    trainXnet = zeros( nF, nTr*nC ); 
    for i = 1:nC
        trainXnet ( :, (i - 1)*nTr + 1 : i*nTr ) = trainIms( :, :, i ); 
    end
    
    % Y data:
    trainYnet = zeros( nC, nTr*nC );     % matlab neural net shaped trainY matrix
    trainY = zeros(nTr*nC, 1 );                 % for svm, nearest neighbors
    % populate Y matrix or vector: 
    for i = 1:nC 
        trainYnet( i, (i-1)*nTr + 1: i*nTr ) = 1;      % 1's in the c'th row
        trainY ( (i-1)*nTr + 1: i*nTr ) = i ; 
    end
    
    %----------------------------------------------------------------------------------------------------------------
    
    % Test:
    
    % X matrices with pixel features only:
    testXnet = zeros( nF, nTe*nC ); 
    for i = 1:nC
        testXnet ( :, (i - 1)*nTe + 1 : i*nTe ) = testIms( :, :, i ); 
    end 
    
    % Y data:
    testYnet = zeros( nC, nTe*nC );     % matlab neural net shaped trainY matrix
    testY = zeros(nTe*nC, 1 );                 % for svm, nearest neighbors
    % populate Y matrix or vector: 
    for i = 1:nC 
        testYnet( i, (i-1)*nTe + 1: i*nTe ) = 1;      % 1's in the c'th row
        testY ( (i-1)*nTe + 1: i*nTe ) = i ; 
    end
    %----------------------------------------------------------------------------------------------------------
    
        
    % do the run
    counter = counter + 1;

    %trainX = trainImsX;
   %valX = testImsX;
    % trainYnet, trainY are already defined 
    valY = testY;               % since we change syntax from 'test' to 'val' (silly). col vector
    valYnet = testYnet; 

    %trainXnet = trainX';
    valXnet = testXnet;  % valX';

    %% fill in some fields in results:

    results(counter).omniPretrainResultsFilename = fullfile(resultsFolder, resultsFilename);
    results(counter).useIms = 1;
    results(counter).useEns = 0; % useNN; since this meant using extra features
    results(counter).enReps = 0; % N/A      enReps;
    results(counter).enWeight = 0; % N/A    enWeight; 
    results(counter).nTr = nTr;
    results(counter).runNum = ind; 


    %% Neural Net:

    type = 'Neural net';

    numHiddenUnits = nF;

    %% Pretrain NN weights on Omniglot: 
     % we have 137 classes, but we want just 10. Choose 10 individual classes at random, and train on just 20 samples per class. 
     selectedClasses = randperm( size(fA_full,3) , nC); % 1:10; %  [89, 123:131]; % for option 2
     omni = zeros(size(fA_full, 1), 20, nC);
     for i = 1:nC
         s = selectedClasses(i); % index of selected class 
         omni( :, : , i ) = fA_full(:, :, s  );
     end 
    %  % show the samples' vectors, by class:
    %  figure,  for i = 1:nC,   subplot(1,10,i),   imshow( omni( :,: , i ) ),  end

    omniTrainRatio = 0.8;
    nTr_omni = size(omni,2);

    % First arrange data and labels for NN consumption:

    % train X matrices with pixel features only:
    omniXnet = zeros( nF, nTr_omni*nC ); 
    for i = 1:nC
        omniXnet ( :, (i - 1)*nTr_omni + 1 : i*nTr_omni ) = omni( :, :, i ); 
    end 

    % Y train data:
    omniYnet = zeros( nC, nTr_omni*nC );     % matlab neural net shaped trainY matrix 
    omniY = zeros( nTr_omni*nC, 1 );
    % populate Y matrix or vector: 
    for i = 1:nC 
        omniYnet( i, (i-1)*nTr_omni + 1: i*nTr_omni ) = 1;      % 1's in the c'th row
        omniY ( (i-1)*nTr_omni + 1: i*nTr_omni ) = i ;  
    end

    netOmni = patternnet(numHiddenUnits);

    netOmni.divideParam.trainRatio = trainRatio;
    netOmni.divideParam.valRatio = 1 - trainRatio;
    netOmni.divideParam.testRatio = 0;
    netOmni.trainParam.showWindow = false;
    % train the net:
    [netOmni, tr ] = train( netOmni, omniXnet, omniYnet );

    % see the accuracy on omni train set as a check:
    yHatProbs = netOmni(omniXnet); 
    yHatOmni = zeros( size(omniY) );
    for i = 1:length(omniY)
        yHatOmni(i) = find( yHatProbs(:,i) == max( yHatProbs(:,i) ) );
    end 
    omniAcc = round(100* sum(yHatOmni == omniY) / length(omniY) );
    % disp( ['omni accuracy = ' num2str(round(omniAcc)) ] )

    %----------------------------------------------------------------------------------------

    % Now run MNIST, with and without pretrained weights

  %% 1. Get an accuracy using imported pre-trained weights from netOmni:
     
    net = patternnet(numHiddenUnits);
    temp = rand(size(trainXnet));  % to ensure linearly independent. 'configure' was choosing too few hidden units for low nTr.
    net = configure(net, temp , trainYnet);  % need to do this to create the spots to import into

    net.iw{1,1} = netOmni.iw{1,1};
    net.lw{2,1} = netOmni.lw{2,1};
    net.b{1,1} = netOmni.b{1,1};
    net.b{2,1} = netOmni.b{2,1};

    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = 1 - trainRatio;
    net.divideParam.testRatio = 0;
    net.trainParam.showWindow = false;

    % train the net:
    [net, tr ] = train( net, trainXnet, trainYnet );

    % classify the validation set:
    yHatProbs = net(valXnet);

    yHat = zeros(size(valY));

    for i = 1:length(valY)
        yHat(i) = find( yHatProbs(:,i) == max( yHatProbs(:,i) ) );
    end

    % Accuracy:
    overallAcc = round(100* sum(yHat == valY)/length(valY) );
    for i = classLabels
        inds = find(valY == classLabels(i));
        classAcc(i) = round(100*sum( yHat(inds) == valY(inds) )/length(valY(inds) ) );
    end
     % save this result in easy form for later console display:
    omniPretrainNNAcc = overallAcc; 
    results(counter).omniPretrainNNAcc = overallAcc;

   %% 2.  get a baseline, with random initial weights:
     
    net = patternnet(numHiddenUnits); 

    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = 1 - trainRatio;
    net.divideParam.testRatio = 0;
    net.trainParam.showWindow = false;

    % train the net:
    [net, tr ] = train( net, trainXnet, trainYnet );

    % classify the validation set:
    yHatProbs = net(valXnet);

    yHat = zeros(size(valY));

    for i = 1:length(valY)
        yHat(i) = find( yHatProbs(:,i) == max( yHatProbs(:,i) ) );
    end

    % Accuracy:
    overallAcc = round(100* sum(yHat == valY)/length(valY) );
    for i = classLabels
        inds = find(valY == classLabels(i));
        classAcc(i) = round(100*sum( yHat(inds) == valY(inds) )/length(valY(inds) ) );
    end
     % save this result in easy form for later console display:
    baselineNNAcc = overallAcc; 

    results(counter).baselineNNAcc = overallAcc;
        
    % display this cyborg's stats:
    disp( [ 'Run #', num2str(ind)  ': ', num2str(nTr), ' numTrain'  ] )
   
    disp( ['     Neural Net: omniAcc, Base, w/omni preTrain = ', num2str(omniAcc), ', ', num2str(baselineNNAcc),....
    ', ', num2str(omniPretrainNNAcc),  '. Rel gain =  ', num2str(round(100*(omniPretrainNNAcc - baselineNNAcc)/ baselineNNAcc )),'%'] ) 
   
    
end % for ind 
end % for numTrain
if ~isfolder(resultsFolder)
    mkdir(resultsFolder)
end
save(fullfile(resultsFolder, resultsFilename), 'results')

% MIT license:
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
% associated documentation files (the "Software"), to deal in the Software without restriction, including 
% without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
% copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to 
% the following conditions: 
% The above copyright notice and this permission notice shall be included in all copies or substantial 
% portions of the Software.
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
% INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
% PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
% COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN 
% AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.




