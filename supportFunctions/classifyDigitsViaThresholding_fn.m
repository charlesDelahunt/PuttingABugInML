
function [output] = classifyDigitsViaThresholding_fn(results, homeAdvantage, homeThresholdSigmas, aboveHomeThreshReward)
% Classify the test digits in a run using log likelihoods from the various EN responses, with the added option of 
% rewarding high scores relative to an ENs home-class expected response distribution.
% One use of this function is to apply de-facto thresholding on discrete ENs, so that the predicted class 
% corresponds to the EN that spiked most strongly (relative to its usual home-class response).
% Inputs:
%  1. results = 1 x 10 struct produced by viewENresponses. i'th entry gives results for all classes, in the i'th EN.
%   Important fields: 
%     a. postMeanResp, postStdResp (to calculate post-training, ie val, digit response distributions).
%     b. postTrainOdorResponse (gives the actual responses for each val digit, for that EN)
%         Note that non-post-train odors have response = -1 as a flag.
%     c. odorClass: gives the true labels of each digit, 1 to 10 (10 = '0'). this is the same in each EN.
%  2.  'homeAdvantage' is the emphasis given to the home EN. It
%           multiplies the off-diagonal of dist. 1 -> no advantage
%           (default). Very high means that a test digit will be classified
%           according to the home EN it does best in, ie each EN acts on
%           it's own.
%  3.  'homeThresholdSigmas' = the number of stds below an EN's home-class mean that we set a threshold, such that 
%           if a digit scores above this threshold in an EN, that EN will
%           be rewarded by 'aboveHomeThreshReward' (below)
%  4.  'aboveHomeThreshReward': if a digit's response scores above the EN's mean home-class  value, reward it by
%           dividing by aboveHomeThreshReward. This reduces the log likelihood score for that EN.    
% Output: 
%    A struct with the following fields:
%    1. likelihoods = n x 10 matrix, each row a postTraining digit. The entries are summed log likelihoods.
%    2. trueClasses = shortened version of whichOdor (with only postTrain, ie validation, entries)
%    3. predClasses = predicted classes
%    4. confusionMatrix = raw counts, rows = ground truth, cols = predicted
%    5. classAccuracies = 1 x 10 vector, with class accuracies as percentages
%    6. totalAccuracy = overall accuracy as percentage

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%---------------------------------------------

% plan:
% 1. for each test digit (ignore non-postTrain digits), for each EN, calculate the # stds from the test
%    digit is from each class distribution. This makes a 10 x 10 matrix
%    where each row corresponds to an EN, and each column corresponds to a class.
% 2. Square this matrix by entry. Sum the columns. Select the col with the lowest value as the predicted
%    class. Return the vector of sums in 'likelihoods'.
% 3. The rest is simple calculation

% the following values of argin2,3,4 correspond to the log likelihood
% classifier in 'classifyDigitsViaLogLikelihood.m':
%     homeAdvantage = 1;
%     homeThresholdSigmas = any number;
%     aboveHomeThreshReward = 1;
% The following value enables pure home-class thresholding:
%     homeAdvantage = 1e12;        % to effectively  eliminate off-diagonals

r = results;
nEn = length(r);   % number of ENs, same as number of classes
ptInds = find(r(2).postTrainOdorResp >= 0);  % indices of post-train (ie validation) digits
nP = length(ptInds);   % number of post-train digits 

% extract true classes:
temp = [r(1).odorClass];     % throughout, digits may be referred to as odors or 'odor puffs'
trueClasses = temp(ptInds);

% extract the relevant odor puffs: Each row is an EN, each col is an odor puff
for i = 1:nEn
    temp = r(i).postTrainOdorResp;
    ptResp(i,:) = temp(ptInds);
end

% make a matrix of mean Class Resps and stds. Each row is an EN, each col is a class. 
% For example, the i'th row, j'th col entry of 'mu' is the mean of the i'th
% EN in response to digits from the j'th class; the diagonal contains the
% responses to the home-class.
for i = 1:nEn
    mu(i,:) = r(i).postMeanResp;
    sig(i,:) = r(i).postStdResp;
end

% for each EN: 
% get the likelihood of each puff (ie each col of ptResp)
likelihoods = zeros(nP,nEn);
for i = 1:nP
    dist = ( repmat(ptResp(:,i), [ 1, nEn ]) - mu )./ sig ;      % 10 x 10 matrix. The ith row, jth col entry is the mahalanobis distance of 
                    % this test digit's response from the i'th ENs response to the j'th class. For example, the diagonal contains
                    % the mahalanobis distance of this digit's response to each EN's home-class response.
    
    % 1. Apply rewards for above-threshold responses:
    offDiag = dist - diag(diag(dist));    
    onDiag =  diag(dist) ; 
    % Reward any onDiags that are above some threshold (mu - n*sigma) of an EN. 
    % CAUTION: This reward-by-shrinking only works when off-diagonals are
    % demolished by very high value of 'homeAdvantage'.
    homeThreshs = homeThresholdSigmas*diag(sig);
    aboveThreshInds = find(onDiag > homeThreshs);
    onDiag(aboveThreshInds) = onDiag(aboveThreshInds) / aboveHomeThreshReward;
    onDiag = diag(onDiag);  % turn back into a matrix
    % 2. Emphasize the home-class results by shrinking off-diagonal values.  This makes the off-diagonals less important in 
    %     the final likelihood sum. This is shrinkage for a different purpose than in the lines above.
    dist = offDiag / homeAdvantage + onDiag; 
    likelihoods(i,:) = sum(dist.^4);    % the ^4 (instead of ^2) is a sharpener. 
                                                       % In pure thresholding case (ie off-diagonals ~ 0), this does not matter.
end

% make predictions:
for i = 1:nP
    predClasses(i) = find(likelihoods(i,:) == min(likelihoods(i,:) ) );
end

% calc accuracy percentages:
for i = 1:nEn
    classAccuracies(i) = 100*sum(predClasses == i & trueClasses == i)/sum(trueClasses == i);
end
totalAccuracy = 100*sum(predClasses == trueClasses)/length(trueClasses);

% confusion matrix:
% i,j'th entry is number of test digits with true label i that were predicted to be j.
conf = confusionmat(trueClasses, predClasses);

output.homeAdvantage = homeAdvantage;
output.trueClasses = trueClasses;
output.predClasses = predClasses;
output.likelihoods = likelihoods;
output.accuracyPercentages = classAccuracies;
output.totalAccuracy = totalAccuracy;
output.confusionMatrix = conf;
output.homeAdvantage = homeAdvantage;
output.homeThresholdSigmas = homeThresholdSigmas;
output.aboveHomeThreshReward = aboveHomeThreshReward;


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










