
% runMothLearnerOnReducedMnist:
%
% Main script to train a moth brain model on a crude (downsampled) MNIST set.
% The moth can be generated from template or loaded complete from file. 
%
% Preparation:
%   1.  Modify 'specifyModelParamsMnist_fn' with the desired parameters for
%        generating a moth (ie neural network), or specify a pre-existing 'modelParams' file to load.
%   2. Edit USER ENTRIES
%
% Order of events:
%   1. Load and pre-process dataset
%   Within the loop over number of simulations:
%       2. Select a subset of the dataset for this simulation (only a few samples are used).
%       3. Create a moth (neural net). Either select an existing moth file, or generate a new moth in 2 steps: 
%           a) run 'specifyModelParamsMnist_fn' and
%               incorporate user entry edits such as 'goal'.
%           b) create connection matrices via 'initializeConnectionMatrices_fn'
%       4. Load the experiment parameters.
%       5. Run the simulation with 'sdeWrapper_fn'
%       6. Plot results, print results to console
%
% Dependencies: Matlab, Statistics and machine learning toolbox, Signal processing toolbox

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%-------------------------------------------------

close all
clear  

%% USER ENTRIES:

useExistingConnectionMatrices = 0;   % if = 1, load 'matrixParamsFilename', which includes filled-in connection matrices
                                                           % if = 0, generate new moth from template in specifyModelParamsMnist_fn.m
matrixParamsFilename = 'sampleMothModelParams'; % struct with all info, including connection matrices, of a particular moth.

numRuns = 1;   % how many runs you wish to do with this moth or moth template, each run using random draws from the mnist set.

goal  = 15; % defines the moth's learning rates, in terms of how many training samples per class give max accuracy. So "goal = 1" gives a very fast learner.
			% if goal == 0, the rate parameters defined the template will be used as-is. if goal > 1, the rate parameters will be updated, even in a pre-set moth.
                        
trPerClass =  3; % the number of training samples per class 
numSniffs = 2;      % number of exposures each training sample 

% Flags to show various images:
showAverageImages = 1;  % to show thumbnails in 'examineClassAveragesAndCorrelations_fn'   
showThumbnailsUsed =  10;   %  N means show N experiment inputs from each class. 0 means don't show any. 
showENPlots = [1, 1];  % 1 to plot, 0 to ignore
% arg1 above refers to statistical plots of EN response changes. One image (with 8 subplots) per EN. 
% arg2 above refers to EN timecourses. Three scaled ENs timecourses on each of 4 images (only one EN on the 4th image).

% To save results if wished:
resultsFilename = 'results';  % will get the run number appended to it.
saveResultsDataFolder = [ ]; % String. If non-empty, 'resultsFilename' will be saved here.
saveResultsImageFolder = []; % String. If non-empty, images will be saved here (if showENPlots also non-zero).

%-----------------------------------------------

%% Misc book-keeping 

classLabels = 1:10;  % For MNIST. '0' is labeled as 10

valPerClass = 15;  % number of digits used in validation sets and in baseline sets

% make a vector of the classes of the training samples, randomly mixed:
trClasses = repmat( classLabels, [ 1, trPerClass ] );
trClasses = trClasses (randperm( length( trClasses ) ) ); 
% repeat these inputs if taking multiple sniffs of each training sample:
trClasses = repmat( trClasses, [ 1, numSniffs ] ) ;


% Experiment details for 10 digit training:
experimentFn = @setMnistExperimentParams_fn;                 

%-----------------------------------

%% Load and preprocess the dataset.

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
indPoolForTrain = 101:300; 
indPoolForPostTrain =  301:400;

% Population preprocessing pools of indices:
preP.indsToAverageGeneral = 551:1000;
preP.indsToCalculateReceptiveField = 551:1000;
preP.maxInd = max( [ preP.indsToCalculateReceptiveField,  indPoolForTrain ] );   % we'll throw out unused samples.

% 2. Pre-processing parameters for the thumbnails:
preP.downsampleRate = 2;
preP.crop = 2;
preP.numFeatures =  85;  % number of pixels in the receptive field
preP.pixelSum = 6;
preP.showAverageImages = showAverageImages; 
preP.downsampleMethod = 1; % 0 means sum square patches of pixels. 1 means use bicubic interpolation.

preP.classLabels = classLabels; % append
preP.useExistingConnectionMatrices = useExistingConnectionMatrices; % append
preP.matrixParamsFilename = matrixParamsFilename;

% generate the data array:
 [ fA, activePixelInds, lengthOfSide ] = generateDownsampledMnistSet_fn(preP); % argin = preprocessingParams

% The dataset fA is a feature array ready for running experiments. Each experiment uses a random draw from this dataset.
% fA = n x m x 10 array where n = #active pixels, m = #digits 
%   from each class that will be used. The 3rd dimension gives the class, 1:10   where 10 = '0'.

%-----------------------------------

% Loop through the number of simulations specified:
disp( [ 'starting sim(s) for goal = ', num2str(goal), ', trPerClass = ', num2str(trPerClass), ', numSniffsPerSample = ' , num2str(numSniffs) ,':' ] )

for run = 1:numRuns  
    
    %% Subsample the dataset for this simulation:

    % Line up the images for the experiment (in 10 parallel queues)
	digitQueues = zeros(size(fA));
    
    for i = classLabels        
        % 1. Baseline (pre-train) images: 
            % choose some images from the baselineIndPool:
            rangeTopEnd = max(indPoolForBaseline) - min(indPoolForBaseline) + 1; 
            % select random digits:
            theseInds = min(indPoolForBaseline) + randsample( rangeTopEnd, valPerClass ) - 1;  % since randsample min pick = 1
            digitQueues( :, 1:valPerClass, i ) = fA(:, theseInds, i );
        
        % 2. Training images:
            % choose some images from the trainingIndPool:
            rangeTopEnd = max(indPoolForTrain) - min(indPoolForTrain) + 1; 
            theseInds = min(indPoolForTrain) + randsample( rangeTopEnd, trPerClass ) - 1;
            % repeat these inputs if taking multiple sniffs of each training sample:
            theseInds = repmat(theseInds, [ numSniffs , 1 ] );
            digitQueues(:, valPerClass + 1:valPerClass + trPerClass*numSniffs, i) = fA(:, theseInds, i) ;
            
        % 3. Post-training (val) images: 
             % choose some images from the postTrainIndPool:
            rangeTopEnd = max(indPoolForPostTrain) - min(indPoolForPostTrain) + 1; 
            % pick some random digits:
            theseInds = min(indPoolForPostTrain) + randsample( rangeTopEnd, valPerClass ) - 1;
            digitQueues(:,valPerClass + trPerClass*numSniffs + 1: valPerClass + trPerClass*numSniffs + valPerClass, i) = fA(:, theseInds, i);
    end % for i = classLabels
    
    % % show the final versions of thumbnails to be used, if wished:
    if showThumbnailsUsed
        tempArray = zeros( lengthOfSide, size(digitQueues,2), size(digitQueues,3));
        tempArray(activePixelInds,:,:) = digitQueues;  % fill in the non-zero pixels
        titleString = 'Input thumbnails'; 
        normalize = 1;
        showFeatureArrayThumbnails_fn(tempArray, showThumbnailsUsed, normalize, titleString );  
                                                                             %argin2 = number of images per class to show.
    end
     
    %----------------------------------------- 
    
    %% Create a moth. Either load an existing moth, or create a new moth:
    
    if useExistingConnectionMatrices
		load( matrixParamsFilename )
	else   % case: new moth
 	    % a) load template params with specify_params_fn:
   		modelParams = specifyModelParamsMnist_fn( length(activePixelInds), goal  );  % modelParams = struct
        
        % c) Now populate the moth's connection matrices using the modelParams:
        modelParams = initializeConnectionMatrices_fn(modelParams);
    end 
	
    modelParams.trueClassLabels = classLabels;     % misc parameter tagging along

	% Define the experiment parameters, including book-keeping for time-stepped evolutions, eg
    %       when octopamine occurs, time regions to poll for digit responses, windowing of Firing rates, etc
    experimentParams = experimentFn( trClasses, classLabels, valPerClass );

    %-----------------------------------
    
    %% 3. run this experiment as sde time-step evolution:

    simResults = sdeWrapper_fn( modelParams, experimentParams, digitQueues );   
    
    %-----------------------------------

    %% Experiment Results: EN behavior, classifier calculations: 
    
    if ~isempty(saveResultsImageFolder)
        if ~exist(saveResultsImageFolder)
            mkdir(saveResultsImageFolder)
        end
    end
    % Process the sim results to group EN responses by class and time:
    r = viewENresponses_fn( simResults, modelParams, experimentParams, ...
                                            showENPlots, classLabels, resultsFilename, saveResultsImageFolder );
    
    % Calculate the classification accuracy:
    % for baseline accuracy function argin, substitute pre- for post-values in r:
    rNaive = r;     
    for i = 1:length(r)
        rNaive(i).postMeanResp = r(i).preMeanResp;
        rNaive(i).postStdResp = r(i).preStdResp;
        rNaive(i).postTrainOdorResp = r(i).preTrainOdorResp;
    end
    
    % 1. Using Log-likelihoods over all ENs:
    %     Baseline accuracy: 
    outputNaiveLogL = classifyDigitsViaLogLikelihood_fn ( rNaive );
    % disp(  'LogLikelihood: ')
        disp( [ 'Naive  Accuracy: ' num2str(round(outputNaiveLogL.totalAccuracy)),...
         '%, by class: ' num2str(round(outputNaiveLogL.accuracyPercentages)),    ' %.   ' ])

    %    Post-training accuracy using log-likelihood over all ENs:
    outputTrainedLogL = classifyDigitsViaLogLikelihood_fn ( r );  
    disp([ 'Trained Accuracy: ' num2str(round(outputTrainedLogL.totalAccuracy)),...
        '%, by class: ' num2str(round(outputTrainedLogL.accuracyPercentages)),    ' %.   '  resultsFilename, '_', num2str(run) ])

    % 2 Using single EN thresholding:
    outputNaiveThresholding = classifyDigitsViaThresholding_fn ( rNaive, 1e9, -1, 10 );
    outputTrainedThresholding = classifyDigitsViaThresholding_fn ( r, 1e9, -1, 10 ); 
%     disp( 'Thresholding: ')
%     disp( [ 'Naive accuracy: ' num2str(round(outputNaiveThresholding.totalAccuracy)),...
%               '%, by class: ' num2str(round(outputNaiveThresholding.accuracyPercentages)),    ' %.   ' ])
%     disp([ ' Trained accuracy: ' num2str(round(outputTrainedThresholding.totalAccuracy)),...
%               '%, by class: ' num2str(round(outputTrainedThresholding.accuracyPercentages)),    ' %.   ' ])

    % append the accuracy results, and other run data, to the first entry of r:
    r(1).modelParams = modelParams;  % will include all connection weights of this moth
    r(1).outputNaiveLogL = outputNaiveLogL;
    r(1).outputTrainedLogL = outputTrainedLogL;
    r(1).outputNaiveThresholding = outputNaiveThresholding;
    r(1).outputTrainedThresholding = outputTrainedThresholding;
    r(1).matrixParamsFilename = matrixParamsFilename;
    r(1).K2Efinal = simResults.K2Efinal;

    if ~isempty(saveResultsDataFolder) 
        if ~exist(saveResultsDataFolder, 'dir' )
            mkdir(saveResultsDataFolder)
        end  
        save( fullfile(saveResultsDataFolder, [resultsFilename, '_', num2str(run) ]) , 'r')
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


