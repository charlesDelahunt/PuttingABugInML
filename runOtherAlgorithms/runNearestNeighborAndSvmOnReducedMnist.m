
% RunNearestNeigborAndSvmOnReducedMnist.m
%
% Main script to train nearest neighbor and SVM classifiers on the reduced MNIST, varying the number of training  samples.
% This applies the same pre-processing as 'runMothLearnerOnReducedMnist', to produce an MNIST-like (but cruder) task.
%
% Dependencies: Matlab, Statistics and machine learning toolbox

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%-----------------------------------------------------------------

close all
clear

%% USER ENTRIES:

numRuns = 5;   % how many runs you wish to do, each run using random draws from the mnist set.

trPerClass =  3; % the number of training samples per class. Do not go above 4000.

% Nearest Neighbors:
runNearestNeighbors = true;
numNeighbors =  1; %  Optimization param for NN. Suggested values:
                                %  trPerClass -> numNeighbors:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5
% SVM:
runSVM = true;
boxConstraint = 1e1;  % Optimization parameter for svm. Suggested values:
                                    % trPerClass -> boxConstraint:  1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 10 -> 1e-1, 20 -> 1e-4 or 1e-5,
                                    % 50 -> 1e-5 ; 100+ -> 1e-7

showAverageImages = 0;
showThumbnailsUsed = 15; % 0 means don't show the experiment inputs. N > 0 means show N digits per class 

%---------------------------------------------------------------------------------------------------------------------

classLabels = 1:10;  % For MNIST. '0' is labeled as 10
valPerClass = 15;   % number of digits per class in the baseline and val sets

%% Load and preprocess the dataset, same as for moth simulations.

% The dataset:
% Because the moth brain architecture, as evolved, only handles ~60 features, we need to
% create a new, MNIST-like task but with many fewer than 28x 28 pixels-as-features.
% We do this by cropping and downsampling the mnist thumbnails, then selecting a subset of the 
% remaining pixels.
% This results in a cruder dataset (set various view flags to see thumbnails).
% However, it is sufficient for testing the moth brain's learning ability. Other ML methods need  
% to be tested on this same cruder dataset to make useful comparisons.

% Define train and control pools for the experiment, and determine the receptive field.
% This is done first because the receptive field determines the number of AL units, which
%      must be updated in modelParams before 'initializeMatrixParams_fn' runs.
% This dataset will be used for each simulation in numRuns. Each
%      simulation draws a new set of samples from this set.

% Parameters:
% Parameters required for the dataset generation function are attached to a struct preP.
% 1. The images used. This includes pools for mean-subtraction, baseline, train, and val. 
%     This is NOT the number of training samples per class. That is trPerClass, defined above. 

% specify pools of indices from which to draw baseline, train, val sets.
indPoolForBaseline = 1:100;
indPoolForTrain =  101:300; 
if trPerClass > 20                                          % if training on lots of images, add to the training pool.
    indPoolForTrain = [101:300, 1001:5400 ];
end
indPoolForPostTrain =   301:400;

% Population preprocessing pools of indices:
preP.indsToAverageGeneral = 551:1000;
preP.indsToCalculateReceptiveField = 551:1000;
preP.maxInd = max( [ preP.indsToCalculateReceptiveField, indPoolForTrain ] );  

% 2. Pre-processing parameters for the thumbnails:
preP.downsampleRate = 2;
preP.crop = 2;
preP.numFeatures =  85;  % number of pixels in the receptive field
preP.pixelSum = 6;
preP.downsampleMethod = 1;  % 0 means sum square patches of pixels. 1 means use bicubic interpolation.

preP.classLabels = classLabels; % append
preP.useExistingConnectionMatrices = 0; % append
preP.showAverageImages = showAverageImages; % append 

% generate the data array:
 [ fA, activePixelInds, lengthOfSide ] = generateDownsampledMnistSet_fn(preP); % argin = preprocessingParams

% The dataset fA is a feature array ready for running experiments. Each experiment uses a random draw from this dataset.
% fA = n x m x 10 array where n = #active pixels, m = #digits per class (for all purposes)
%   from each class that will be used. The 3rd dimension gives the class, 1:10 where 10 = '0'.

%-----------------------------------
% Loop through the number of simulations specified:
disp( [ 'starting classifier(s)...' ] )

for run = 1:numRuns

    %% Subsample the dataset for this simulation:

    % Line up the images for the experiment (in 10 parallel queues)
    digitQueues = zeros(size(fA));

    for i = classLabels
        % 1. Baseline (pre-train) images (not used in these methods):
        %     choose random images from the baselineIndPool:
        rangeTopEnd = max(indPoolForBaseline) - min(indPoolForBaseline) + 1;
        theseInds = min(indPoolForBaseline) + randsample( rangeTopEnd, valPerClass ) - 1;  % since randsample min pick = 1
        digitQueues( :, 1:valPerClass, i ) = fA(:, theseInds, i );

        % 2. Training images:
        %     choose some images from the trainingIndPool:
        rangeTopEnd = max(indPoolForTrain) - min(indPoolForTrain) + 1;
        theseInds = min(indPoolForTrain) +  randsample( rangeTopEnd, trPerClass ) - 1;
        digitQueues(:, valPerClass + 1:valPerClass + trPerClass, i) = fA(:, theseInds, i) ;

%             % for one-shot scenario  (not relevant for these methods)
%             if oneShot
%                 theseInds(:) = theseInds(1);
%             end

        % 3. Post-training (Val) images:
        %     choose some images from the postTrainIndPool:
        rangeTopEnd = max(indPoolForPostTrain) - min(indPoolForPostTrain) + 1;
        % pick some random digits:
        theseInds = min(indPoolForPostTrain) +  randsample( rangeTopEnd, valPerClass ) - 1;
        digitQueues(:, valPerClass + trPerClass + 1: valPerClass + trPerClass + valPerClass, i) = fA(:, theseInds, i);
    end % for i = classLabels

     % % show the final versions of thumbnails to be used, if wished:
    if showThumbnailsUsed
        tempArray = zeros( lengthOfSide, size(digitQueues,2), size(digitQueues,3));
        tempArray(activePixelInds,:,:) = digitQueues ;  % fill in the non-zero pixels
        titleString = 'Input thumbnails'; 
        normalize = 1;
        showFeatureArrayThumbnails_fn(tempArray, showThumbnailsUsed, normalize, titleString );    
                                                                             %argin2 = number of images per class to show. 
    end

    %% Re-organize train and val sets for classifiers:

    % Build train and val feature matrices and class label vectors.
    % X = n x numberPixels;  Y = n x 1, where n = 10*trPerClass. 
    trainX = zeros( 10*trPerClass, size(fA,1) );
    valX = zeros( 10*valPerClass, size(fA,1) );
    trainY = zeros( 10*trPerClass, 1 );
    valY = zeros( 10*valPerClass, 1 );
    
    dQ = digitQueues;  % shorthand for the actual images used in this run
    
    % Populate these one class at a time:
    for i = classLabels
        % Skip the first 'valPerClass' digits, as these are used as baseline digits in the moth (formality).
        temp = dQ( :, valPerClass + 1:valPerClass + trPerClass, i );   
        trainX( (i-1)*trPerClass + 1: i*trPerClass , : ) = temp';
        temp = dQ( :, valPerClass + trPerClass + 1:valPerClass + trPerClass + valPerClass, i );
        valX( (i-1)*valPerClass + 1: i*valPerClass , : ) = temp';
        
        trainY( (i-1)*trPerClass + 1: i*trPerClass  ) = i ;
        valY( (i-1)*valPerClass + 1: i*valPerClass , : ) = i ;
    end
    
    %% NEAREST NEIGHBORS:

    if runNearestNeighbors

        % Use matlab built-in function.
        % Optimizations:
        %   1. Standardize features
        %   2. Number of neighbors
         
        type = 'Nearest neighbor'; 
        
        nnModel = fitcknn(trainX, trainY,'NumNeighbors', numNeighbors, 'Standardize', 1 );

        [yHat ] = predict(nnModel, valX );

        % Accuracy:
        overallAcc = round(100* sum(yHat == valY) / length(valY) );
        for i = classLabels
            inds = find(valY == classLabels(i));
            classAcc(i) = round(100*sum( yHat(inds) == valY(inds) ) / length(valY(inds) ) );
        end
        disp( [ type,  ': ', num2str(trPerClass), ' training samples per class. ',...
            ' Accuracy = ', num2str(overallAcc), '%. numNeigh = ', num2str(numNeighbors), ...
            '. Class accs (%): ', num2str(classAcc) ] )        
    end
    %-------------------------------------------------------------------------------

    %% SVM:
    if runSVM

        % Create an SVM template, ie let matlab do the work.
        % Note: This can take a long time to run.
        % Optimizations:
        % 1. Standardize features
        % 2. BoxConstraint parameter

        type = 'SVM';
        t = templateSVM('Standardize', 1,  'BoxConstraint',  boxConstraint); % , 'KernelFunction','gaussian');  % gaussian kernel did not work as well
        svmModel = fitcecoc( trainX, trainY, 'Learners', t , 'FitPosterior', 1);    %  'OptimizeHyperparameters','auto');
                                                                            % Newer versions of matlab have an 'OptimizeHyperparameters' option

        [yHat, negLoss, pbScore, posteriors ] = predict(svmModel, valX);  

        % Accuracy:
        overallAcc = round(100* sum(yHat == valY)/length(valY) );
        for i = classLabels
            inds = find(valY == classLabels(i));
            classAcc(i) = round(100*sum( yHat(inds) == valY(inds) )/length(valY(inds) ) );
        end
        disp( [ type ': ' num2str(trPerClass) ' training samples per class. ',...
            ' Accuracy = ' num2str(overallAcc) '%.',  ' BoxConstraint = ' num2str(boxConstraint),...
            '. Class accs (%): ' num2str(classAcc) ] )
  
    end

end % for run 


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