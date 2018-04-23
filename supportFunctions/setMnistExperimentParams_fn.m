
function [expParams]  = setMnistExperimentParams_fn( trClasses, classLabels, valPerClass ) 
%
% This function defines parameters of a time-evolution experiment: overall timing, stim timing and strength, octo timing and strength, 
%         lowpass window parameter,  etc.
% It does book-keeping to allow analysis of the SDE time-stepped evolution of the neural firing rates.
% Inputs:
%       1. trClasses: vector of indices giving the classes of the training digits in order. 
%                    The first entry must be nonzero. Unused entries can be filled with -1s if wished.
%       2. classLabels: a list of labels, eg 1:10 for mnist
%       3. valPerClass: how many digits of each class to use for baseline and post-train
% Output:
%   1. expParams: struct with experiment info.

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%----------------------------------------------------------------

% Order of time periods:
%       1. no event period: allow system to settle to a steady state spontaneous FR baseline
%       2. baseline period: deliver a group of digits for each class
%       3. no event buffer 
%       5. training period:  deliver digits + octopamine + allow hebbian updates 
%       6. no event buffer 
%       7. post-training period: deliver a group of digits for each class

stimMag = 20; % (stim magnitudes as passed into AL. See original version in smartAsABug codebase)
stimLength = 0.22;

nC = length(classLabels);            % the number of classes in this experiment

%% Define the time span and events:

step = 3;                     % the time between digits (3 seconds)
trStep = step + 2;      % allow more time between training digits  

expParams.simStart = -30;   % use negative start-time for convenience (artifact)

% Baseline period:
% do a loop, to allow gaps between class groups:
baselineTimes = [];  
startTime = 30;
gap = 10;
for i = 1:nC
    baselineTimes = [ baselineTimes, startTime : step : startTime + (valPerClass - 1)*step  ];% vector of timepoints, one digit applied every 'step' seconds
    startTime = max(baselineTimes) + gap;
end 
endOfBaseline = max(baselineTimes) + 25;   % include extra buffer before training

% Training period:
trainTimes = endOfBaseline : trStep : endOfBaseline + (length(trClasses) - 1)*trStep;      % vector of timepoints, one digit every 'trStep' seconds
endOfTrain = max(trainTimes) + 25 ;   % includes buffer before Validation

% Val period:
% do a loop, to allow gaps between class groups: 
valTimes = [];
startTime = endOfTrain;
for i = 1:nC
    valTimes = [ valTimes, startTime : step : startTime + (valPerClass - 1)*step  ]; % vector of timepoints 
    startTime = max(valTimes) + gap;
end
endOfVal = max(valTimes) + 4;  

%% assemble vectors of stimulus data for export:

% Assign the classes of each stim. Assign the baseline and val in blocks, and the training stims in the order passed in:
whichClass = zeros( size ( [baselineTimes, trainTimes, valTimes ] ) );
numBaseline = valPerClass*nC;
numTrain = length(trClasses);
for c = 1:nC
    whichClass( (c-1)*valPerClass + 1: c*valPerClass)  = classLabels(c);  % the baseline groups
    whichClass( numBaseline + numTrain + (c-1)*valPerClass + 1 :  numBaseline + numTrain + c*valPerClass )  = classLabels(c);  % the val groups
end
whichClass( numBaseline + 1:numBaseline + numTrain ) = trClasses;
expParams.whichClass = whichClass;

stimStarts =  [ baselineTimes, trainTimes, valTimes ]; 
expParams.stimStarts = stimStarts; % starting times
expParams.durations = stimLength*ones( size( stimStarts ) );      % durations
expParams.classMags = stimMag*ones( size( stimStarts ) );              % magnitudes 

% octopamine input timing:
expParams.octoMag = 1; 
expParams.octoStart = trainTimes;   
expParams.durationOcto = 1;

% heb timing: Hebbian updates are enabled 25% of the way into the stimulus, and
% last until 75% of the way through (ie active during the peak response period)
expParams.hebStarts = trainTimes + 0.25*stimLength;
expParams.hebDurations = 0.5*stimLength*ones(size(trainTimes));
expParams.startTrain = min(expParams.hebStarts);
expParams.endTrain = max(expParams.hebStarts) + max(expParams.hebDurations);

%% Other time parameters required for time evolution book-keeping:

% the numbers 1,2,3 do refer to time periods where spont responses are allowed to settle before recalibration.
expParams.startPreNoiseSpontMean1 = -25;     
expParams.stopPreNoiseSpontMean1 = -15;
% Currently no change is made in start/stopSpontMean2. So spontaneous behavior may be stable in this range.
expParams.startSpontMean2 = -10;
expParams.stopSpontMean2 = -5;
% currently, spontaneous behavior is steady-state by startSpontMean3.
expParams.startSpontMean3 = 0;
expParams.stopSpontMean3 = 28;

expParams.preHebPollTime = min(trainTimes) - 5;
expParams.postHebPollTime = max(trainTimes) + 5;

% timePoints for plotting EN responses:
% spontaneous response periods, before and after training, to view effect of training on spontaneous FRs:
expParams.preHebSpontStart = expParams.startSpontMean3;
expParams.preHebSpontStop = expParams.stopSpontMean3;
expParams.postHebSpontStart = max(trainTimes) + 5;
expParams.postHebSpontStop = min(valTimes) - 3;

% hamming filter window parameter (= width of transition zone in seconds). The lp filter is applied to odors and to octo
expParams.lpParam =  0.12;

expParams.simStop = max(stimStarts) + 10;


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
