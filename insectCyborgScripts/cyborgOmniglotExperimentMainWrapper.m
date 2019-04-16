% Main wrapper to run insect cyborg experiments. as described in the paper, using Omniglot as a dataset

% 'Insect cyborgs: Biological feature generators improve machine accuracy on limited data'
% To use this wrapper:
% 1. Set User Entries in the three scripts listed below (the first script controls number of runs and numTrain).
% 2. Set directories and result filenames in the three scripts listed below (so they all match)
% 3. Run this script

% Dependencies: Matlab, Statistics and machine learning toolbox, Signal processing toolbox
% Copyright (c) 2019 Charles B. Delahunt
% MIT License

clear

% ATTENTION! By default this does a shortened set of runs:  numRuns = 5, trPerClassList = [ 3, 7 ]. 
% For reproduce full data from the paper, in the first script set variables: numRuns = 13, trPerClass = [ 1 2 3 5 7 10 15 20 30 40 50 70 100 ].

runMothLearnerOnOmniglotForUseInCyborg
       % ( calls 'generateDownsampledOmniglotSet_normalizeByMaxVal_fn.m' )
runCyborgLearnersOnOmniglot
plotCyborgResults


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
