
% Apply ML methods with and without top-10 PLS modes of training data as extra features:
% This is easier than running MothNet cyborgs, because we don't need to do
% time-evolutions to train the feature generator as we do with MothNet.
%
% Order of events:
% 0. Select training and test data
% 1. Apply PCA (really SVD) to training, select top 10 modes. Use projection onto these modes to 
%     generate features for both train and val sets 
% 2. Train and test various ML methods (NN, SVM, Nearest Neighbor) on the training and
%     val sets  (standard ML, as baseline)
% 2. Trainand test the same ML methods on the same training and val sets, but with
%     the 10 PCA projections added as additional features. 

% Dependencies: Matlab, Statistics and machine learning toolbox, Signal processing toolbox
% Copyright (c) 2019 Charles B. Delahunt
% MIT License

close all
clear

counter = 0;

%% USER ENTRIES:

resultsFolder = 'comparisonFeatureGeneratorsResultsFolder';

% ATTENTION! USING THE SAME RESULTSFILENAME WILL OVERWRITE PREVIOUS RESULTS!
resultsFilename = 'withPlsFeatures_1';  % results for all runs will be saved in one .mat file 

% Specify which baseline ML methods to use:
runNearestNeighbors = 1;
runSVM = 1;
runNeuralNet = 1;  

% Specify which numTrain values to run:
numTrainList = [ 1 2 3 5 7 10 15 20 30 40 50 70 100 ];  % loop through these
numRuns = 13;

% END OF USER ENTRIES

%----------------------------------------------------------------------------------------------------------

% Some other parameters that do not need to be tweaked to run the code:

meanSubtractFlag = 1;   % for pca
 
nTe = 21;        % numTest. Test = Holdout (used interchangeably)
nF  = 85;               % number of pixel features 
 
% Weight the 10 PCA modes relative to the 85 pixel features:
enReps =1;       % repeat EN outputs as features 'enWeight' times. 3 is good for Nearest Neigh and 
                          %  SVM. Should have no effect on NNs
enWeight = 10; % factor difference between median PCA feature vals and median non-zero pixel vals. 
                          % No effect on Near, svm due to standardization.
                          % 10 is good for Neural nets

showThumbnailsUsed = 0;
nC = 10;  % number of classes
classLabels = 1:nC;

% Method-specific parameters:
% Nearest Neighbors:
numNeighbors =  1; %  Optimization param for NN. Suggested values:
%  nTr -> numNeighbors:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5

% SVM:
boxConstraint = [ 1e4 ];    % this gets updated based on nTr, according to:
                % Optimization parameter for svm. Suggested values:
                % nTr -> boxConstraint:  1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 
                % 10 -> 1e-1; 20 -> 1e-4 or 1e-5; 50 -> 1e-5 ; 100+ -> 1e-7
                
% Neural net:
numHiddenUnits = 85; % placeholder, changes after we see if ENs are being used or not.  % assume one hidden layer
trainRatio = 1.0;   % use all training samples for training (since we have only a few)

%% Generate the dataset from MNIST:
% This is the same as for the MothNet cyborg experiments.
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
for run = 1:numRuns  
    
	if run == 1
    	disp( [ 'starting PLS tests, nTr = ', num2str(nTr) ,':' ] )
	end

	counter = counter + 1;
	%% Subsample the dataset for this simulation:

	% Line up the images for the experiment (in 10 parallel queues)
	digitQueues = zeros(size(fA)); 
   
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
    
   %% Extract PLS coefficients from train ims:
    % fA = n x m x 10 array where n = #active pixels, m = #digits 
    %         from each class that will be used. The 3rd dimension gives the class, 1:10   where 10 = '0'.
    
    % Vars in [ xL, yL, xS ] = plsregress( X, Y, nModes ):
    %       X = nTr x nF, ie rows are samples and cols are features. X must be zero-centered!
    %       Y = nTr x 1,  values 1 to 10.
    %       xL = nF x nModes = the latent components.
    %       xS = nTr x nModes, orthonormal. The i'th col gives the coeffs to construct
    %                the i'th latent component as a linear sum of samples (rows) in X. 
    %                are the PLS features.
    
    % make a matrix of all training data, each row is a sample, each col is a feature:
    plsData = zeros( size(trainIms,1), nTr * nC );
    plsY = zeros(nTr*nC, 1);
    for i = 1:nC
        plsData(:, (i-1)*nTr + 1: i*nTr ) = trainIms(:, :, i );
        plsY( (i-1)*nTr + 1: i*nTr ) = i; 
    end
    plsDataMean = zeros( [size(plsData,1), 1]);  
    if meanSubtractFlag
        plsDataMean = mean(plsData,2);
        plsData = plsData - repmat( plsDataMean, [1, size(plsData,2) ] ); 
    end
    plsData = plsData';  % now nTr x nF, ie rows are samples, cols are features
    
    % do PLS:
    [xL,yL  ] = plsregress(plsData, plsY, nC );
    % xL = nF x nC matrix = the latent components. We project onto these.   
 
    % make the PCA features
    plsTrainFeat = zeros( nC, nTr*nC );
    plsTestFeat = zeros( nC, nTe*nC );
    for i = 1:nC 
        temp = zeros( nF, nTr);
        temp (:, : ) = trainIms(:, :, i );
        plsTrainFeat( :, (i-1)*nTr + 1: i*nTr ) = xL' * temp; 
        temp = zeros( nF, nTe);
        temp (:, : ) = testIms(:, :, i );
        plsTestFeat( :, (i-1)*nTe + 1: i*nTe ) = xL' * temp;
    end
    % we now have trainIms, testIms, pcaTrainFeat, pcaTestFeat.
    
    %% set up matrices for ML:
    
    % Train:
    % X matrices with PCA features only:
    trainPlsXnet  = plsTrainFeat;   % matlab NN-shaped trainX matrix with PCA
    trainPlsX = trainPlsXnet';     % transpose to make trainX matrix for svm, nearest neighbors
    
    % X matrices with pixel features only:
    trainImsXnet = zeros( nF, nTr*nC ); 
    for i = 1:nC
        trainImsXnet ( :, (i - 1)*nTr + 1 : i*nTr ) = trainIms( :, :, i ); 
    end
    trainImsX = trainImsXnet';    % transpose to make trainX matrix for svm, nearest neighbors
    
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
    % X matrices with PCA features only:
    testPlsXnet  = plsTestFeat;   % matlab NN-shaped trainX matrix with PCA
    testPlsX = testPlsXnet';     % transpose to make trainX matrix for svm, nearest neighbors
    
    % X matrices with pixel features only:
    testImsXnet = zeros( nF, nTe*nC ); 
    for i = 1:nC
        testImsXnet ( :, (i - 1)*nTe + 1 : i*nTe ) = testIms( :, :, i ); 
    end
    testImsX = testImsXnet';    % transpose to make trainX matrix for svm, nearest neighbors  
    
    % Y data:
    testYnet = zeros( nC, nTe*nC );     % matlab neural net shaped trainY matrix
    testY = zeros(nTe*nC, 1 );                 % for svm, nearest neighbors
    % populate Y matrix or vector: 
    for i = 1:nC 
        testYnet( i, (i-1)*nTe + 1: i*nTe ) = 1;      % 1's in the c'th row
        testY ( (i-1)*nTe + 1: i*nTe ) = i ; 
    end
    %----------------------------------------------------------------------------------------------------------
    
    % Run ML methods:
    
    for usePls = [ 0, 1 ]   % ie try with and without PCA features. Always use pixel features 
        
        % do the run
        counter = counter + 1;

        % finish populating trainX etc (we needed useEns):
        trainX = trainImsX;
        valX = testImsX;
        % trainYnet, trainY are already defined 
        valY = testY;               % since we chang syntax from 'test' to 'val' (silly). col vector
        valYnet = testYnet;

        % normalize the EN readings, to get them in the ballpark range of pixel values:
        temp = testImsX(:);
        pixelVal = median( temp(temp > 0) );
        enVal = median( testPlsX(:) );
        valEnXnormed = testPlsX * pixelVal / enVal * enWeight;
        trainEnXnormed = trainPlsX * pixelVal / enVal * enWeight;
        
        if usePls
            for i = 1:enReps
                trainX = [ trainX, trainEnXnormed ];
                valX = [ valX, valEnXnormed ];
            end
        end

        trainXnet = trainX';
        valXnet = valX';

        %% fill in some fields in results:

        results(counter).pcaResultsFilename = fullfile(resultsFolder, resultsFilename);
        results(counter).useIms = 1;
        results(counter).useEns = usePls;
        results(counter).enReps = enReps;
        results(counter).enWeight = enWeight; 
        results(counter).nTr = nTr;
        results(counter).runNum = run; 

        %% NEAREST NEIGHBORS:

        if runNearestNeighbors

            % Use matlab built-in function.
            % Optimizations:
            %   1. Standardize features
            %   2. Number of neighbors. 1 to 10 nTr: 1; 50 nTr: 9;  5000 nTr: 17.
            numNeighbors = 1;
            if nTr >= 20, numNeighbors = 3; end
            if nTr >= 50, numNeighbors = 9; end

            type = 'Nearest neighbor';

            % Try many numNeighbor values:
            nnModel = fitcknn(trainX, trainY,'NumNeighbors', numNeighbors, 'Standardize', 1 );

            [yHat ] = predict(nnModel, valX );

            % Accuracy:
            overallAcc = round(100* sum(yHat == valY) / length(valY) );
            for i = classLabels
                inds = find(valY == classLabels(i));
                classAcc(i) = round(100*sum( yHat(inds) == valY(inds) ) / length(valY(inds) ) );
            end
            % save this result in easy form for later console display: 
            if usePls == 0
                baselineNearNeighAcc = overallAcc;
            else
                withPlsFeatsNearNeighAcc = overallAcc;          % ENs + image pixels
            end 

            results(counter).nearNeighAcc = overallAcc;
        end
        %-------------------------------------------------------------------------------

        %% SVM:
        if runSVM && nTr > 1

            % Create an SVM template, ie let matlab do the work.
            % Note: This can take a long time to run.
            % Optimizations:
            % 1. Standardize features
            % BoxConstraint parameter: 1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 10 -> 1e-1, 20 -> 1e-4
            % modify boxConstraint:
            nTrVals = [ 2 3 5 7 10 15 20 30 40 50 70 100];
            boxConstVals = [ 1e4, 1e4, 1e0, 1e0, 1e-1, 1e-3, 1e-4, 1e-5 1e-5 1e-5 1e-6, 1e-7];
            boxConstraint = boxConstVals( nTrVals == nTr  );

            type = 'SVM';
            t = templateSVM('Standardize', 1,  'BoxConstraint',  boxConstraint); % , 'KernelFunction','gaussian');  % gaussian kernel did not work as well
            svmModel = fitcecoc( trainX, trainY, 'Learners', t , 'FitPosterior', 1);    %  'OptimizeHyperparameters','auto');
            % Newer versions of matlab have an 'OptimizeHyperparameters' option

            [yHat, negLoss, pbScore, posteriors ] = predict(svmModel, valX);    % if using 'FitPosterior', 1: you can add argouts   negLoss, pbScore, posteriors

            % Accuracy:
            overallAcc = round(100* sum(yHat == valY)/length(valY) );
            for i = classLabels
                inds = find(valY == classLabels(i));
                classAcc(i) = round(100*sum( yHat(inds) == valY(inds) )/length(valY(inds) ) );
            end
             % save this result in easy form for later console display: 
            if usePls == 0
                baselineSvmAcc = overallAcc;
            else
                withPlsFeatsSvmAcc = overallAcc;          % ENs + image pixels
            end 

            results(counter).svmAcc = overallAcc;
        end

        %% Neural Net:
        if runNeuralNet

            type = 'Neural net';

            numHiddenUnits = 85 + usePls*enReps*nC;

            nTrNet = trainRatio*nTr;   % because only some get used for training

            % First rearrange data and labels for NN consumption:
            % NOTE: this is now done earlier. Remove one or the other section of code.
            % data: each col = one sample, each row = 1 feature:
            trainXnet = trainX';
            valXnet = valX';
            % labels: 10 x n matrix of 1s and 0s:
            trainYnet = zeros(10, length(trainY));
            for i = 1:length(trainY)
                trainYnet(trainY(i), i) = 1;
            end
            valYnet = zeros(10, length(valY));
            for i = 1:length(valY)
                valYnet(valY(i), i) = 1;
            end

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
            if usePls == 0
                baselineNNAcc = overallAcc;
            else
               withPlsFeatsNNAcc = overallAcc;          % ENs + image pixels
            end 

            results(counter).neuralNetAcc = overallAcc;
        end   % if run neural net
    end % for useEns
    % display this cyborg's stats:
    disp( [ 'Run #', num2str(run)  ': ', num2str(nTr), ' numTrain'  ] )
    if runNearestNeighbors
        disp( ['     Nearest neighbor: Base, PCA = ', num2str(baselineNearNeighAcc),....
        ', ' , num2str(withPlsFeatsNearNeighAcc), '. Rel gain =  ', num2str(round(100*(withPlsFeatsNearNeighAcc - baselineNearNeighAcc)/ baselineNearNeighAcc )),'%'] )    
    end
    if runSVM
        disp( ['     SVM: Base, PCA = ', num2str(baselineSvmAcc),....
        ', ', num2str(withPlsFeatsSvmAcc),  '. Rel gain =  ', num2str(round(100*(withPlsFeatsSvmAcc - baselineSvmAcc)/ baselineSvmAcc )),'%'] ) 
    end
    if runNeuralNet
        disp( ['     Neural Net: Base, PCA = ', num2str(baselineNNAcc),....
        ', ', num2str(withPlsFeatsNNAcc),  '. Rel gain =  ', num2str(round(100*(withPlsFeatsNNAcc - baselineNNAcc)/ baselineNNAcc )),'%'] ) 
    end
    
end % for run 
end % for numTrain
if ~isdir(resultsFolder)
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




