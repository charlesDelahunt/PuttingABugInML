
function [ results ] = viewENresponsesForCyborg_fn( simResults, modelParams, experimentParams,...
    showPlots, classLabels, resultsFilename, saveImageFolder)

% This is a slight modification of 'viewENresponses_fn' for use with cyborgs. The key
% change is that we need to track when the trained moth re-runs training samples, because we 
% need these EN responses as input features when training the cyborg. These re-run samples are 
% stored in 'rerunTrainingOdorResp', which depends on the experimentParam 'rerunTrainingOdorTimes'.
% This fn should work with prior moth code, but the prior 'viewENresponses_fn' does not work with cyborgs.

% View readout neurons (EN):
% Color-code their dots by class and by concurrent octopamine.
% Collect stats: median, mean, and std of FR for each digit, pre- and post-training.
% Throughout, digits may be referred to as odors, or as odor puffs.
% 'Pre' = naive. 'Post' = post-training
%
% Inputs:
%   1. simResults: output of sdeWrapper_fn.m
%   2. modelParams: struct of this moth's parameters
%   3. experimentParams: struct with timing and digit class info from the experiment.
%   4. showPlots: 1 x 2 vector. First entry: show changes in accuracy. 2nd entry: show EN timecourses.
%   5. classLabels: 1 to 10
%   6. resultsFilename:  to generate image filenames if saving. Optional argin
%   7. saveImageFolder:  where to save images. If this = [], images will not be saved (ie its also a flag). Optional argin.
%
% Outputs (as fields of resultsStruct):
%   1. preMeanResp = numENs x numOdors matrix = mean of EN pre-training
%   2. preStdResp = numENs x numOdors matrix = std of EN responses pre-training
%   3. ditto for post etc
%   4. percentChangeInMeanResp = 1 x numOdors vector
%   5. trained = list of indices corresponding to the odor(s) that were trained
%   6. preSpontMean = mean(preSpont);
%   7. preSpontStd = std(preSpont);
%   8. postSpontMean = mean(postSpont);
%   9. postSpontStd = std(postSpont);

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%-----------------------------------------------------

if nargin < 7, saveImageFolder = []; end
if nargin < 6, resultsFilename = []; end

if ~isempty(saveImageFolder)
    if ~exist( saveImageFolder, 'dir')
        mkdir( saveImageFolder )
    end
end

nE = modelParams.nE;

% pre- and post-heb spont stats:
preHebSpontStart = experimentParams.preHebSpontStart;
preHebSpontStop = experimentParams.preHebSpontStop;
postHebSpontStart =  experimentParams.postHebSpontStart;
postHebSpontStop =  experimentParams.postHebSpontStop;

colors = {  [0 0 1], [0 0 0], [1 0 0], [0 1 0], [1 0 1], [ 0 1 1],...
    [1  0.3  0.8], [0.8  0.3  1], [0.8  1  0.3], [0.5 0.5 0.5]  }; %  for 10 classes
% concurrent octopamine will be marked with yellow x's.

E = simResults.E;   % # timesteps x #ENs
T = simResults.T;   % # timesteps x 1
octoHits = simResults.octoHits;
if max(octoHits) > 0
    octoTimes = T(octoHits > 0);
else
    octoTimes = [];
end

% calc spont stats:
preSpont = E( T> preHebSpontStart & T < preHebSpontStop );
postSpont = E( T > postHebSpontStart & T < postHebSpontStop );
preHebMean = mean(preSpont);
preHebStd = std(preSpont);
postHebMean = mean(postSpont);
postHebStd = std(postSpont);

% Set regions to examine:
% 1. data from experimentParams:
simStart = experimentParams.simStart;
classMags = experimentParams.classMags;
stimStarts = experimentParams.stimStarts; % to get timeSteps from very start of sim
stimStarts = stimStarts.*(classMags > 0);  % ie only use non-zero puffs
whichClass = experimentParams.whichClass;
whichClass = whichClass.*(classMags > 0);
startTrain = experimentParams.startTrain;
endTrain = experimentParams.endTrain;
startReRunningTrainingOdors = min( experimentParams.rerunTrainingOdorTimes ) - 5;
% special case if we have only val odors:
if isempty(startReRunningTrainingOdors)
    startReRunningTrainingOdors = experimentParams.simStop;
end

classList = sort(unique(whichClass));
numClasses = length(classList);

% Make one stats plot per EN. Loop through ENs:

for enInd = 1:nE
    thisEnResponse = E(:, enInd );
    %% Calculate pre- and post-train odor response stats:
    % Assumes that there is at least 1 sec on either side of an odor without octo
    
    for i = 1:length(stimStarts)
        t = stimStarts(i);
        % Note: to find no-octo stimStarts, there is a certain amount of machinery in order to mesh with the timing data from the experiment.
        % For some reason octoTimes are not recorded exactly as listed in format short mode. So we need to
        %                              use abs difference > small thresh, rather than ~ismember(t, octoTimes):
        small = 1e-8;
        % assign no-octo, PRE-train response val (or -1):
        preTrainOdorResp(i) = -1;  % as flag
        if isempty(octoTimes)
            preTrainOdorResp(i) = max( thisEnResponse ( T > t-1 & T < t+1 ) );
        elseif min(abs(octoTimes - t)) > small && t < startTrain
            preTrainOdorResp(i) = max( thisEnResponse ( T > t-1 & T < t+1 ) );
        end
        % assign no-octo, POST-train response val (or -1): NOTE: this does
        % not include the set of training odors run through the trained network
        postTrainOdorResp(i) = -1;
        if ~isempty(octoTimes)
            if min(abs(octoTimes - t)) > small && t > endTrain && t < startReRunningTrainingOdors
                postTrainOdorResp(i) = max( thisEnResponse ( T > t-1 & T < t+1 ) );
            end
        end
        if isempty(octoTimes)
            if t > endTrain -5 && t < startReRunningTrainingOdors
                postTrainOdorResp(i) = max( thisEnResponse ( T > t-1 & T < t+1 ) );
            end
        end
        
        % record and tag the training odors that were re-run through the trained network:
        rerunTrainingOdorResp(i)  = -1;
        if t > startReRunningTrainingOdors
            rerunTrainingOdorResp(i) = max( thisEnResponse( T > t - 1 & T < t + 1 ) );
        end
        
    end
    
    % calc no-octo stats for each odor, pre and post train:
    for k = 1:numClasses
        preFull = preTrainOdorResp( preTrainOdorResp >=0 & whichClass == classList(k) );
        postFull = postTrainOdorResp( postTrainOdorResp >=0 & whichClass == classList(k) );
        % calculate the averaged sniffs of each sample: SA means 'sniffsAveraged'
        preSA = zeros(1, length( preFull ) );    % this will contain the average responses over all sniffs for each sample
        for i = 1:length(preSA)
            preSA(i) = mean( preFull( (i-1) + 1 : i ) );
        end
        postSA = zeros( 1, length(postFull) );
        for i = 1:length(postSA)
            postSA(i) = mean( postFull( (i-1) + 1 : i ) );
        end
        
        if isempty(preSA)
            preMeanResp(k) = -1;
            preMedianResp(k) = -1;
            preStdResp(k) = -1;
            preNumPuffs(k) = 0;
        else
            preMeanResp(k) = mean(preSA);
            preMedianResp(k) = median(preSA);
            preStdResp(k) = std(preSA);
            preNumPuffs(k) = length(preSA);
        end
        if isempty(postSA)
            postMeanResp(k) = -1;
            postMedianResp(k) = -1;
            postStdResp(k) = -1;
            postNumPuffs(k) = 0;
        else
            postMeanResp(k) = mean(postSA);
            postMedianResp(k) = median(postSA);
            postStdResp(k) = std(postSA);
            postNumPuffs(k) = length(postSA);
        end
    end % for k
    
    % to plot +/- 1 std of % change in meanResp, we want the std of our
    % estimate of the mean = stdResp/sqrt(numPuffs). Make this calc:
    preStdMeanEst = preStdResp./sqrt(preNumPuffs);
    postStdMeanEst = postStdResp./sqrt(postNumPuffs);
    
    preSA = find(preNumPuffs > 0);
    postSA = find(postNumPuffs > 0);
    postOffset = postSA + 0.25;
    
    % a key output:
    percentChangeInMeanResp = 100*(postMeanResp(preSA) - preMeanResp(preSA) )./preMeanResp(preSA);
    percentChangeInNoiseSubtractedMeanResp =...
        100*(postMeanResp(preSA) - preMeanResp(preSA) - postHebMean)./preMeanResp(preSA);
    
    percentChangeInMedianResp = 100*(postMedianResp(preSA) - preMedianResp(preSA) )./preMedianResp(preSA);
    percentChangeInNoiseSubtractedMedianResp =...
        100*(postMedianResp(preSA) - preMedianResp(preSA) - postHebMean)./preMedianResp(preSA);
    
    % create cell of xticklabels:
    trueXLabels = cell(size(classLabels));
    for j = 1:length(classLabels)
        trueXLabels{j} = num2str(mod(classLabels(j),10) );   % the 'mod' turns 10 into 0
    end
    %% plot stats if wished:
    if showPlots(1)
        
        del = texlabel('Delta');
        scrsz = get(0,'ScreenSize');
        thisFig = figure('Position',[scrsz(1), scrsz(2), scrsz(3)*0.8, scrsz(4)*0.8 ]);
        % medians
        subplot(2,3,1)
        hold on,
        grid on,
        
        % raw median, pre and post:
        plot(preSA, preMedianResp(preSA),'*b')
        plot(postOffset, postMedianResp(postSA),'bo','markerfacecolor','b')
        %     plot(pre, preMeanResp + preStdResp,'+g')
        %     plot(post, postMeanResp + postStdResp,'+g')
        %     plot(pre, preMeanResp - preStdResp,'+g')
        %     plot(post, postMeanResp - postStdResp,'+g')
        
        % make the home EN of this plot red:
        plot(enInd, preMedianResp(enInd), 'ro')
        plot(enInd + 0.25, postMedianResp(enInd), 'ro','markerfacecolor','r')
        title(['EN ' num2str(enInd), ' median +/- std' ])
        xlim([0,max(preSA) + 1])
        xticks(preSA)
        xticklabels(trueXLabels)
        ylim( [0  1.1*max([ preMedianResp, postMedianResp ]) ] )
        
        % connect pre to post with lines for clarity:
        for j = 1:length(preSA)
            lineColor = 'b';
            if j == i, lineColor = 'r'; end
            plot( [ preSA(j),postOffset(j) ], [ preMedianResp(preSA(j)), postMedianResp(preSA(j)) ], lineColor )
        end
        
        % percent change in medians:
        subplot(2,3,2)
        hold on,
        grid on,
        plot(preSA, 100*(postMedianResp(preSA) - preMedianResp(preSA) )./preMedianResp(preSA),...
            'bo','markerfacecolor','b')
        % mark the trained odors in red:
        plot(enInd, 100*(postMedianResp(enInd) - preMedianResp(enInd) )./preMedianResp(enInd),...
            'ro','markerfacecolor','r')
        title( [ '% ' del ' median' ])
        xlim([0,max(preSA) + 1])
        % ylim([-50,400])
        xticks(preSA)
        xticklabels(trueXLabels)
        
        % relative changes in median, ie control/trained:
        subplot(2,3,3)
        pn = sign( postMedianResp(enInd) - preMedianResp(enInd) );
        hold on,
        grid on,
        plot(preSA, pn*( (postMedianResp(preSA) - preMedianResp(preSA) )./preMedianResp(preSA) ) / ...
            ( (postMedianResp(enInd) - preMedianResp(enInd) )./preMedianResp(enInd) ),...
            'bo','markerfacecolor','b')
        % mark the trained odors in red:
        plot(enInd, pn*1, 'ro','markerfacecolor','r')
        title(['relative ' del ' median'])
        xlim([0,max(preSA) + 1])
        % ylim([0,2])
        xticks(preSA)
        xticklabels(trueXLabels)
        
        %------------------------------------------------------------------------
        % means
        % raw means, pre and post:
        subplot(2,3,4)
        hold on,
        grid on,
        errorbar(preSA, preMeanResp(preSA), preStdResp(preSA), 'bo')
        errorbar(postOffset, postMeanResp(postSA), postStdResp(postSA),'bo','markerfacecolor','b')
        errorbar(enInd, preMeanResp(enInd), preStdResp(enInd), 'ro')
        errorbar(enInd + 0.25, postMeanResp(enInd),  postStdResp(enInd), 'ro','markerfacecolor','r')
        title(['EN ' num2str(enInd), ' mean +/- std'])
        xlim([0,max(preSA) + 1])
        xticks(preSA)
        xticklabels(trueXLabels)
        ylim( [0  1.1*max([ preMeanResp, postMeanResp ]) + max([preStdResp, postStdResp]) ] )
        
        % plot spont:
        errorbar(preSA(1), preHebMean, preHebStd,'mo')
        errorbar(postOffset(1), postHebMean, postHebStd,'mo','markerfacecolor','m')
        
        % percent change in means
        subplot(2,3,5)
        hold on,
        grid on,
        plot(preSA, percentChangeInMeanResp,'bo','markerfacecolor','b')
        % mark the trained odors in red:
        plot(enInd, percentChangeInMeanResp(enInd),'ro','markerfacecolor','r')
        title([ '% ' del ' mean' ] )
        xlim([0,max(preSA) + 1])
        % ylim([-50,1000])
        xticks(preSA)
        xticklabels(trueXLabels)
        
        % relative percent changes
        subplot(2,3,6)
        pn = sign( postMeanResp(enInd) - preMeanResp(enInd) );
        hold on,
        grid on,
        plot(preSA, pn*percentChangeInMeanResp  / percentChangeInMeanResp(enInd),'bo','markerfacecolor','b')
        % mark the trained odors in red:
        plot(enInd, pn*1, 'ro','markerfacecolor','r')
        title(['relative ' del ' mean' ])
        xlim([0,max(preSA) + 1])
        % ylim([0,2])
        xticks(preSA)
        xticklabels(trueXLabels)
        
    end % if showPlots
 % Save plot code:
    if ~isempty(saveImageFolder) && showPlots(1)
        saveas( thisFig, fullfile(saveImageFolder,[resultsFilename '_en' num2str(enInd) '.png']), 'png')
    end
    
    %---------------------------------------------------------------------------------
    
    % store results in a struct:
    r(enInd).preTrainOdorResp = preTrainOdorResp;  % preserves all the sniffs for each stimulus
    r(enInd).postTrainOdorResp = postTrainOdorResp;   % validation odors
    r(enInd).rerunTrainingOdorResp = rerunTrainingOdorResp;
    r(enInd).preRespSniffsAved = preSA;   % the averaged sniffs for each stimulus
    r(enInd).postRespSniffsAved = postSA;
    r(enInd).odorClass = whichClass;
    r(enInd).percentChangeInMeanResp = percentChangeInMeanResp;  % key stat
    r(enInd).percentChangeInNoiseSubtractedMeanResp = percentChangeInNoiseSubtractedMeanResp;
    r(enInd).relativeChangeInNoiseSubtractedMeanResp = ...
        percentChangeInNoiseSubtractedMeanResp / percentChangeInNoiseSubtractedMeanResp(enInd);
    r(enInd).percentChangeInMedianResp = percentChangeInMedianResp;
    r(enInd).percentChangeInNoiseSubtractedMedianResp = percentChangeInNoiseSubtractedMedianResp;
    r(enInd).relativeChangeInNoiseSubtractedMedianResp = ...
        ( (postMedianResp - preMedianResp - postHebMean )./preMedianResp ) / ...
        ( (postMedianResp(enInd) - preMedianResp(enInd) - postHebMean )./preMedianResp(enInd) );
    r(enInd).trained = enInd;
    % EN odor responses, pre and post training.
    % these should be vectors of length numStims
    r(enInd).preMeanResp = preMeanResp;
    r(enInd).preStdResp = preStdResp;
    r(enInd).postMeanResp = postMeanResp;
    r(enInd).postStdResp = postStdResp;
    % spont responses, pre and post training:
    r(enInd).preSpontMean = mean(preSpont);
    r(enInd).preSpontStd = std(preSpont);
    r(enInd).postSpontMean = mean(postSpont);
    r(enInd).postSpontStd = std(postSpont);
    
    results = r;
    
end % for enInd = 1:numClasses

%% Plot EN timecourses normalized by mean digit response:

labels = whichClass;

if showPlots(2)
    scrsz = get(0,'ScreenSize');
    % go through each EN:
    for enInd = 1:nE           % recal EN1 targets digit class 1, EN2 targets digit class 2, etc
        if mod(enInd - 1,3) == 0
            enFig2 = figure('Position',[scrsz(1), scrsz(2), scrsz(3)*1, scrsz(4)*1 ]); % make a new figure at ENs 4, 7, 10
        end
        
        subplot(3, 1, mod(enInd - 1, 3) + 1 )
        hold on
        
        xlim([-30, max(T)])
        
        % plot octo
        plot( octoTimes, zeros(size(octoTimes)), 'yx')
        
        % plot mean pre and post training of trained digit:
        preMean = r(enInd).preMeanResp;
        preMeanTr = preMean(enInd);
        preMeanControl = mean(preMean( [ 1:enInd-1, enInd+1:numClasses ]));
        preStd = r(enInd).preStdResp;
        preStd = preStd(enInd);
        postMean = r(enInd).postMeanResp;
        postMeanTr = postMean(enInd);
        postMeanControl = mean(postMean( [ 1:enInd-1, enInd+1:numClasses ]));
        postStd = r(enInd).postStdResp;
        postStd = postStd(enInd);
        preTime = T(T < startTrain);
        preTimeInds = find(T < startTrain);
        postTime = T(T > endTrain);
        postTimeInds = find(T > endTrain);
        midTime = T(T > startTrain & T < endTrain);
        midTimeInds = find(T > startTrain & T < endTrain);
        
        % plot ENs:
        
        plot( preTime, E(preTimeInds,enInd)/preMeanControl , 'b');    % normalized by the home class preMean
        plot( postTime, E(postTimeInds,enInd)/postMeanControl, 'b');    % normalized by the home class postMean
        plot( midTime, E(midTimeInds, enInd)/ 1, 'b')
        
        %             plot(preTime, preMean*ones(size(preTime)), 'color', colors{enInd},'lineStyle','-')
        %             plot(postTime, postMean*ones(size(postTime)), 'color', colors{enInd},'lineStyle','-')
        %             plot(preTime, (preMean-preStd)*ones(size(preTime)), 'color', colors{enInd},'lineStyle',':')
        %             plot(postTime, (postMean-postStd)*ones(size(postTime)), 'color', colors{enInd},'lineStyle',':')
        % plot stims by color
        for i = 1:numClasses
            plot( stimStarts(whichClass == classList(i)), zeros(size(stimStarts(whichClass == classList(i)))),...
                'color', colors{i},'marker','.' ,'markersize',24, 'markerfacecolor', colors{i} ),
            % reinforce trained color:
            if i == enInd
                plot( stimStarts(whichClass == classList(i)),...
                    0.001*ones(size(stimStarts(whichClass == classList(i)))),...
                    'color', colors{i},'marker','.' ,'markersize',24, 'markerfacecolor', colors{i} ),
            end
        end
        
        set(gca,'fontname', 'Arial','fontweight','bold','fontsize',12)
        % format:
        ylim( [ 0, 1.2* max(E(postTimeInds,enInd))/postMeanControl ] )
        rarrow = texlabel('/rarrow');
        title([ 'EN ' num2str(enInd) ' for class ' num2str( enInd ) ])
        
 % Save EN timecourse:
        if ~isempty(saveImageFolder)  && (mod(enInd, 3) == 0 || enInd == 10)
            saveas( enFig2, fullfile(saveImageFolder,[resultsFilename '_enTimecourses' num2str(enInd) '.png']), 'png')
        end

    end  % for enInd = 1:nE
    
end % if showPlots


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




















