% Apply ML methods to either ENs only, or to pixels and ENs:
% Uses the output of moths trained in 'runMothLearnerOnROmniglotForUseAsCyborg.m'
% For each moth, this script:
% 1. Trains and tests various ML methods (NN, SVM, Nearest Neighbor) on the training and
%     val sets used for that moth (standard ML, as baseline)
% 2. Trains and tests the same ML methods on the same training and holdout sets, but with
%     the trained moth's 10 readout neurons used as additional features (these are the cyborgs).
%     Since we have trained moth responses to train and val sets, we can use the moth
%     readouts as features during both training and testing.

% Dependencies: Matlab, Statistics and machine learning toolbox, Signal processing toolbox
% Copyright (c) 2019 Charles B. Delahunt
% MIT License

close all
clear

counter = 0;

%% USER ENTRIES:

numFeatures = 120; % NOTE! This must agree with 'numFeatures' in 'runMothLearnerOnOmniglotForUseInCyborg.m'
parentFolder = 'mothsForOmniglotCyborg';   % folder containing files saved by 'runMothLearnerOnOmniglotForUseInCyborg.m' 
resultsFilename = 'cyborgOmniglotResults';  % results for all cyborgs will be saved in one .mat file. It is saved into the current dir.
%                                                                      It must match the value in 'plotCyborgOmniglotResults.m'

% END OF USER ENTRIES

%----------------------------------------------------------------------------------------------------------

% Some other parameters that do not need to be tweaked to run the code:

% Weight the 10 moth readouts relative to the 85 pixel features:
enReps = 3;       % repeat EN outputs as features 'enWeight' times. 3 is good for Nearest Neigh and 
                          %  SVM, 1 is best for Neural nets (but should have no effect)
enWeight = 10; % factor difference between median en feature vals and median non-zero pixel vals. 
                          % No effect on Near, svm due to standardization.
                          % 10, 100 are good for Neural nets

showThumbnailsUsed = 0;
nC = 10;  % number of classes

% Nearest Neighbors:
runNearestNeighbors = 1;
numNeighbors =  1; %  Optimization param for NN. Suggested values:
%  trPerClass -> numNeighbors:  1,3,5 -> 1;  (10, 20, 50) -> 1 or 3;  100 -> 3; 500 + -> 5
% SVM:
runSVM = 1;
boxConstraint = [ 1e4 ];    % this gets updated based on trPerClass, according to:
                % Optimization parameter for svm. Suggested values:
                % trPerClass -> boxConstraint:  1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 
                % 10 -> 1e-1; 20 -> 1e-4 or 1e-5; 50 -> 1e-5 ; 100+ -> 1e-7
% Neural net:
runNeuralNet = 1;
numHiddenUnits = 85; % placeholder, changes after we see if ENs are being used or not.  % assume one hidden layer

trainRatio = 1.0;   % use all training samples for training (since we have only a few)


%% Start processing:

classLabels = 1:nC;

% load the results files:
files= dir(parentFolder);
for i = 1:length(files)
    keep(i) = ~isempty(strfind(files(i).name, 'goal'));
end
files = files(keep == 1);

% process each file in turn:

for ind = 1:length(files)
    f = files(ind);
    % extract key params:
    name = f.name;
    sniffs = strfind(name,'_sniffs');
    numSniffs = str2num(name(sniffs + 7));
    tr1 = strfind(name,'_tr');
    tr2 = strfind(name, '_sniffs');
    trPerClass = str2num(name(tr1 + 3:tr2 - 1));
    run1 = strfind(name, '_run');
    run2 = strfind(name, '_');
    run2 = run2(6);     % ie the 6th underscore in the filename
    runNum = str2num(name(run1 + 4:run2 - 1));
    trivialAL = strfind(name,'_trivialAL');
    trivialAL = str2num(name(trivialAL + 10));
    
    % modify numNeighbors if trPerClass is high:
    if trPerClass > 20
        numNeighbors = 3;
    end
    
    % inds of train and val images:
    load( fullfile( parentFolder, name) )  % loads r
    odorClass = r(1).odorClass;
    % selectedClassInds = r(1).selectedClassInds;
    rerunTrainInds = find(r(1).rerunTrainingOdorResp > -1 );
    valInds = find(r(1).postTrainOdorResp > - 1 );
    valPerClass = length(valInds) / nC;
    
    nF = r(1).modelParams.nF;
    
    trainEnXnet  = zeros(nC, length(rerunTrainInds) );   % EN outputs, matlab neural net shaped trainX matrix
    trainEnYnet = zeros( nC, length(rerunTrainInds ) );     % matlab neural net shaped trainY matrix
    trainEnY = zeros(size(rerunTrainInds));    % for svm, nearest neighbors
    % populate these:
    for i = 1:length(rerunTrainInds)
        for c = 1:nC
            trainEnXnet(c, i)  = r(c).rerunTrainingOdorResp( rerunTrainInds(i) );
        end
        trainEnYnet( odorClass( rerunTrainInds(i) ), i ) = 1;
        trainEnY (i) = odorClass( rerunTrainInds(i) );
    end
    trainEnX = trainEnXnet';     % transpose for svm, nearest neighbors
    
    % extract val EN responses the same way:
    valEnXnet  = zeros(nC, length(valInds) );   % EN outputs, matlab neural net shaped trainX matrix
    valEnYnet = zeros( nC, length(valInds ) );     % matlab neural net shaped trainY matrix
    valEnY = zeros(size(valInds));    % for svm, nearest neighbors
    % populate these:
    for i = 1:length(valInds)
        for c = 1:nC
            valEnXnet( c, i )  = r( c ).postTrainOdorResp( valInds(i) );
        end
        valEnYnet( odorClass( valInds(i) ), i ) = 1;
        valEnY (i) = odorClass( valInds(i) );
    end
    valEnX = valEnXnet';     % transpose for svm, nearest neighbors
    
    % extract the images' pixel values from r:
    trainImBlock = r(1).trainIms;   % nF x trPerClass x 10 array
    valImBlock = r(1).valIms;        % nF x valPerClass x 10 array
    
    trainImsXnet = zeros( nF, trPerClass*nC );
    trainImsYnet = zeros( nC, trPerClass*nC );
    trainImsY = zeros(1, trPerClass*nC);
    for c = 1:nC
        trainImsXnet ( :, (c - 1)*trPerClass + 1 : c*trPerClass ) = trainImBlock( :, :, c );
        trainImsYnet ( c, (c - 1)*trPerClass + 1 : c*trPerClass ) = 1;
        trainImsY( (c - 1)*trPerClass + 1 : c*trPerClass  ) = c;
    end
    trainImsX = trainImsXnet';
    
    valImsXnet = zeros( nF, valPerClass*nC );
    valImsYnet = zeros( nC, valPerClass*nC );
    valImsY = zeros(1, valPerClass*nC);
    for c = 1:nC
        valImsXnet ( :, (c - 1)*valPerClass + 1 : c*valPerClass ) = valImBlock( :, :, c );
        valImsYnet ( c, (c - 1)*valPerClass + 1 : c*valPerClass ) = 1;
        valImsY( (c - 1)*valPerClass + 1 : c*valPerClass  ) = c;
    end
    valImsX = valImsXnet';
    
    if showThumbnailsUsed
        tempArray = zeros( 12^2, size(trainImBlock,2), size(trainImBlock,3));
        tempArray(r(1).activePixelInds,:,:) = trainImBlock;  % fill in the non-zero pixels
        titleString = 'Input thumbnails';
        normalize = 1;
        showFeatureArrayThumbnails_fn(tempArray, showThumbnailsUsed, normalize, titleString );
        %argin2 = number of images per class to show.
    end
    
    mothAcc = round(r(1).outputTrainedLogL.totalAccuracy);
    
    %----------------------------------------------------------------------------------------------------------
    
    % Run ML methods:
    
    for useIms = [ 1, 0 ]
        for useEns = [ 0, 1 ]
            
            if  ~useIms && ~useEns
                % case: no features at all. skip
            else
                % do the run
                counter = counter + 1;
                %  disp( [ 'ims = ' num2str(useIms), ', ens = ', num2str(useEns),  ':' ] )
                
                % finish populating trainX etc (we needed useIms and useEns):
                trainX = [];
                valX = [];
                trainYnet =  trainImsYnet;  % same as trainEnsYnet
                trainY = trainImsY;   % same as trainEnsY
                valY = valImsY;
                valYnet = valImsYnet;
                
                % normalize the EN readings, to get them in the ballpark range of pixel values:
                temp = valImsX(:);
                pixelVal = median( temp(temp > 0) );
                enVal = median( valEnX(:) );
                valEnXnormed = valEnX * pixelVal / enVal * enWeight;
                trainEnXnormed = trainEnX * pixelVal / enVal * enWeight;
                if useIms
                    trainX = [ trainX, trainImsX ];
                    valX = [ valX, valImsX ];
                end
                if useEns
                    for i = 1:enReps
                        trainX = [ trainX, trainEnXnormed ];
                        valX = [ valX, valEnXnormed ];
                    end
                end
                
                trainXnet = trainX';
                valXnet = valX';
                valY = valY';   % make into a column
                
                %% fill in some fields in results:
                
                results(counter).mothResultsFilename = fullfile(parentFolder, name);
                results(counter).useIms = useIms;
                results(counter).useEns = useEns;
                results(counter).enReps = enReps;
                results(counter).enWeight = enWeight;
                results(counter).trivialAL = trivialAL;
                results(counter).trPerClass = trPerClass;
                results(counter).runNum = runNum;
                results(counter).mothAcc = r(1).outputTrainedLogL.totalAccuracy;
                
                %% NEAREST NEIGHBORS:
                
                if runNearestNeighbors
                    
                    % Use matlab built-in function.
                    % Optimizations:
                    %   1. Standardize features
                    %   2. Number of neighbors. 1 to 10 trPerClass: 1; 50 trPerClass: 9;  5000 trPerClass: 17.
                    numNeighbors = 1;
                    if trPerClass >= 20, numNeighbors = 3; end
                    if trPerClass >= 50, numNeighbors = 9; end
                    
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
                    if useEns == 0
                        baselineNearNeighAcc = overallAcc;
                    else
                        if useIms == 1
                            cyborg95NearNeighAcc = overallAcc;          % ENs + image pixels
                        else 
                            cyborg10NearNeighAcc = overallAcc;          % ENs only
                        end
                    end 
                    
                    results(counter).nearNeighAcc = overallAcc;
                end
                %-------------------------------------------------------------------------------
                
                %% SVM:
                if runSVM && trPerClass > 1
                    
                    % Create an SVM template, ie let matlab do the work.
                    % Note: This can take a long time to run.
                    % Optimizations:
                    % 1. Standardize features
                    % BoxConstraint parameter: 1 -> NA; 3 -> 1e4; 5 -> 1e0 or 1e1; 10 -> 1e-1, 20 -> 1e-4
                    % modify boxConstraint:
                    trPerClassVals = [ 2 3 5 7 10 15 20 30 40 50 70 100];
                    boxConstVals = [ 1e4, 1e4, 1e0, 1e0, 1e-1, 1e-3, 1e-4, 1e-5 1e-5 1e-5 1e-6, 1e-7];
                    boxConstraint = boxConstVals( trPerClassVals == trPerClass  );
                    
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
                    if useEns == 0
                        baselineSvmAcc = overallAcc;
                    else
                        if useIms == 1
                            cyborg95SvmAcc = overallAcc;          % ENs + image pixels
                        else 
                            cyborg10NSvmAcc = overallAcc;          % ENs only
                        end
                    end 
                    
                    results(counter).svmAcc = overallAcc;
                end
                
                %% Neural Net:
                if runNeuralNet
                    
                    type = 'Neural net';
                    
                    numHiddenUnits = useIms*85 + useEns*enReps*nC;
                    
                    trPerClassNet = trainRatio*trPerClass;   % because only some get used for training
                    
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
                    if useEns == 0
                        baselineNNAcc = overallAcc;
                    else
                        if useIms == 1
                            cyborg95NNAcc = overallAcc;          % ENs + image pixels
                        else 
                            cyborg10NNAcc = overallAcc;          % ENs only
                        end
                    end 
                    
                    results(counter).neuralNetAcc = overallAcc;
                end   % if run neural net
            end % if-else   ~useIms && ~useEns
        end % for useEns
    end % for useIms
    % display this cyborg's stats:
    disp( [ 'Moth #', num2str(ind)  ': ', num2str(trPerClass), ' tr per class. Moth accuracy (%) = ', ...
        num2str(mothAcc) ] )
    disp( ['     Nearest neighbor: Baseline, Cyborg = ', num2str(baselineNearNeighAcc),....
        ', ' , num2str(cyborg95NearNeighAcc), '. Relative gain =  ', num2str(round(100*(cyborg95NearNeighAcc - baselineNearNeighAcc)/ baselineNearNeighAcc )),'%'] )    
    disp( ['     SVM: Baseline, Cyborg = ', num2str(baselineSvmAcc),....
        ', ', num2str(cyborg95SvmAcc),  '. Relative gain =  ', num2str(round(100*(cyborg95SvmAcc - baselineSvmAcc)/ baselineSvmAcc )),'%'] ) 
    disp( ['     Neural Net: Baseline, Cyborg = ', num2str(baselineNNAcc),....
        ', ', num2str(cyborg95NNAcc),  '. Relative gain =  ', num2str(round(100*(cyborg95NNAcc - baselineNNAcc)/ baselineNNAcc )),'%'] ) 
    
end % for ind (of file)
save(resultsFilename, 'results')  % this results file is saved in the current dir

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




