% plotCyborgResults.m:
% plots the accuracy results of cyborgs (and pure moths, and pure ML methods) run and saved by 'runCyborgLearnersOnReducedMnist.m'
% Basically the same as 'plotCyborgResults.m'
%
% Dependencies: Matlab, Statistics and machine learning toolbox, Signal processing toolbox
% Copyright (c) 2019 Charles B. Delahunt
% MIT License

clear

%--------------------------------------------------------------
%% USER ENTRIES:

normalAL = true; % Choose whether to plot normal or pass-through AL cyborgs 
useOnlyEnsInCyborgs = false;  % if true, the ML-cyborg uses only the moth ENs as features, and ignores the image pixels. 

resultsFilename = 'cyborgMnistResults'; % saved by 'runCyborgLearnersOnReducedMnist.m' in current dir.

plotLog = false;

% END USER ENTRIES
%--------------------------------------------------------------------------------------

%% Extract data:
disp([ 'loading ' resultsFilename ])  
load(resultsFilename)  % loads struct = 'results', the results for all the runs done by 'runCyborgLearnersOnReducedMnist.m' 

if isfield(results,'trPerClass')
    trList = sort(unique([results.trPerClass])); % the different 'training samples per class' cases found in results
else 
    trList = [ 1 2 3 5 7 10 15 20 30 40 50  70 100 ];  % 10 may
end

if normalAL
	alType =  [results.trivialAL] == 0; 
	alTag = 'normal AL';
else
	alType =  [results.trivialAL] == 1;  
	alTag = 'pass-through AL';
end

% Feature indicators:
en = [results.useEns] == 1;    % whether the particular result used the moth ENs as features (0 = pure ML, 1 = cyborg)
ims = [results.useIms] == 1;   % whether the particular result used the image pixels as features (0 = moth ENs only, 1 = use pixels)

for i = 1:length(trList)
    tr = [results.trPerClass] == trList(i);
    
    % Case: Use ENs only: the '01' in the variable names means 'yes moth', 'no images'. 
    % The ML method uses only the moth EN output as features, and ignores the original pixel values.
    a = results( ~ims & en & tr & alType );
    % order the results in cols: moth, nearest, svm, net
    med01(i, 1) = median([a.mothAcc]);                 
    med01(i,2) = median([a.nearNeighAcc]);
    med01(i, 3) = median([a.svmAcc]);
    med01(i, 4) = median([a.neuralNetAcc]);
    mu01(i, 1) = mean([a.mothAcc]);
    mu01(i,2) = mean([a.nearNeighAcc]);
    mu01(i, 3) = mean([a.svmAcc]);
    mu01(i, 4) = mean([a.neuralNetAcc]);
    std01(i, 1) = std([a.mothAcc]);
    std01(i,2) = std([a.nearNeighAcc]);
    std01(i, 3) = std([a.svmAcc]);
    std01(i, 4) = std([a.neuralNetAcc]);
    
    % svm did not run for tr = 1:
    if min(trList) == 1
        med01(1,3) = 10;
        mu01(1,3) = 10;
        std01(1,3) = 0;
    end
    
    % use pixels only:      
	% Case: Pure ML: the moth output is not used. This gives the ML baseline.
    a = results( ims & ~en & tr & alType );
    % order the results in cols: moth, nearest, svm, net
    med10(i, 1) = median([a.mothAcc]);
    med10(i,2) = median([a.nearNeighAcc]);
    med10(i, 3) = median([a.svmAcc]);
    med10(i, 4) = median([a.neuralNetAcc]);
    mu10(i, 1) = mean([a.mothAcc]);
    mu10(i,2) = mean([a.nearNeighAcc]);
    mu10(i, 3) = mean([a.svmAcc]);
    mu10(i, 4) = mean([a.neuralNetAcc]);
    std10(i, 1) = std([a.mothAcc]);
    std10(i,2) = std([a.nearNeighAcc]);
    std10(i, 3) = std([a.svmAcc]);
    std10(i, 4) = std([a.neuralNetAcc]);
    % svm did not run for tr = 1:
    if min(trList) == 1
        med10(1,3) = 10;
        mu10(1,3) = 10;
        std10(1,3) = 0;
    end
    
    % use both ens and pixels:
	% Case: The ML method uses both pixels and trained moth outputs as features. This gives the cyborg results.
    a = results( ims & en & tr & alType );
    % order the results in cols: moth, nearest, svm, net
    med11(i, 1) = median([a.mothAcc]);
    med11(i,2) = median([a.nearNeighAcc]);
    med11(i, 3) = median([a.svmAcc]);
    med11(i, 4) = median([a.neuralNetAcc]);
    mu11(i, 1) = mean([a.mothAcc]);
    mu11(i,2) = mean([a.nearNeighAcc]);
    mu11(i, 3) = mean([a.svmAcc]);
    mu11(i, 4) = mean([a.neuralNetAcc]);
    std11(i, 1) = std([a.mothAcc]);
    std11(i,2) = std([a.nearNeighAcc]);
    std11(i, 3) = std([a.svmAcc]);
    std11(i, 4) = std([a.neuralNetAcc]);
    
    % svm did not run for tr = 1, so fill in results as for random guessing:
    if min(trList) == 1
        med11(1,3) = 10;
        mu11(1,3) = 10;
        std11(1,3) = 0; 
    end
    
    % measure gains by sample, pixels only to ens plus pixels, ie pure ML to cyborg:
    % 'a' above is ens + pixels
    b = results( ims & ~en & tr & alType );  % pixels only
    baselineAcc(i,2) = mean( [b.nearNeighAcc] );
    baselineAcc(i,3) = mean( [b.svmAcc] );
    baselineAcc(i,4) = mean( [b.neuralNetAcc] );
    meanOfGains(i, 2) = mean( 100*( [a.nearNeighAcc] - [b.nearNeighAcc] )./ [b.nearNeighAcc]  );
    meanOfGains(i, 3) = mean( 100*( [a.svmAcc] - [b.svmAcc] )./ [b.svmAcc] );
    meanOfGains(i, 4) = mean( 100*( [a.neuralNetAcc] - [b.neuralNetAcc] )./ [b.neuralNetAcc] );
    
end

% Define the first and second results to plot. Default is pure ML -> cyborg.

% The first set of results to plot is pure ML, ie '10':
start = mu10;
stdStart = std10;
startTag = '10';

% The second set of results are for the cyborgs, either ENs + images (default) or ENs only (image pixels ignored)
final = mu11;
stdFinal = std11;
finalTag = '11';
if useOnlyEnsInCyborgs    % case: ignore the image pixels when training cyborgs
	final = mu01;
	stdFinal = std01; 
	finalTag = '01';
end

gainOfMeans = 100*(final - start)./start;   % these are gains, comparing means (not the mean of individual gains) 
% note that these are N x 4, where N = number of nTr values, and each col
%     is an ML method: 1 = moth alone (ie a dummy in terms of ML gains), 2 = Nearest, 3 = SVM, 4 = NN.

% calculate the relative reduction in error:
startErr = 100 - start; 
rawReductionInErr = final - start; % should be positive in most cases
relReductionInErr = 100*rawReductionInErr./startErr;

% calculate fisher linear discriminant and p-values:
fD = ( final - start  ) ./ ( 0.5*stdFinal  + 0.5*stdStart  );
fD = fD';
pVals = 100*(1 - erf( fD / sqrt(2) ) );  % As percentage. Also: erf assumes std = sqrt(2), so we must scale
pVals = min(pVals, 100*ones(size(pVals)));

% print out the mean of the relative gains:
disp('Mean relative gains due to pretraining NN as %, by numTrain:' )
disp([ 'trList: ', num2str(trList) ] )
disp('row 1 = nearest, row 2 = SVM, row 3 = neural net:')
disp(num2str( ( round( (meanOfGains(:,2:4) )' ,1 ) ) ) )


% print out relative gain, comparing mean of all runs, 
% ie how much bigger is the mean of the 13 cyborg runs relative to the mean of the 13 baseline runs:
disp('Gain in mean accuracies due to pretraining NN as %, by numTrain:' )
disp('ie how much bigger is the mean of the cyborg runs relative to the mean of the baseline runs:')
disp( [ 'trList: ', num2str(trList) ] )
disp('row 1 = nearest, row 2 = SVM, row 3 = neural net:')
disp(num2str( ( round( (gainOfMeans(:,2:4) )' ,1 ) ) ) )

% print significance of gain as fisher discrim and as p-values:
disp('fisher discriminants: ' )
disp('row 1 = nearest, row 2 = SVM, row 3 = neural net')
disp( [ 'trList: ', num2str(trList) ] )
disp( num2str(round( fD(2:4, : ) , 1) ) )
disp('p-values:')
disp('row 1 = nearest, row 2 = SVM, row 3 = neural net')
disp(num2str(trList))
disp(num2str(round( pVals(2:4, : ), 0 )))

% print out relative reduction in error, comparing mean of all runs, 
% ie how much smaller is the mean error of the 13 cyborg runs relative to the mean error of the 13 baseline runs:
disp('Gain in mean accuracies due to pretraining NN as %, by numTrain:' )
disp('ie how much bigger is the mean of the cyborg runs relative to the mean of the baseline runs:')
disp( [ 'trList: ', num2str(trList) ] )
disp('row 1 = nearest, row 2 = SVM, row 3 = neural net:')
disp(num2str( ( round( (relReductionInErr(:,2:4) )' ,1 ) ) ) )
%--------------------------------------------------------------------- 
%--------------------------------------------------------------------- 
   
%% Make plots 

colors = {'k', 'g', 'b', 'r'};  
step = 0.5;  % to separate ML methods

% 1. of Means:
figure
    hold on
    % moth:
    plot(trList , start(:,1), colors{1} )
    % ML methods:
    order = [3 2 4];
    for ind = 1:3
        i = order(ind);       
        plot(trList + (ind-1)*step , start(:, i),  [colors{i} 'o'], 'markersize', 8  );
    end   
    for ind = 1:3
        i = order(ind);
        plot(trList   + (ind-1)*step , final(:, i),   [colors{i} 'o'], 'markersize', 11 );
    end
    for ind = 1:3
        i = order(ind);
        for j = 1:length(trList)
            line([trList(j) + (ind-1)*step   , trList(j)   + (ind-1)*step  ], [start(j,i), final(j,i) ], 'color', colors{i}, 'linewidth',3 )
        end
    end
    if plotLog
        set(gca, 'XScale', 'log')
    end
    xlabel('# training samples per class')
    ylabel('Percent accuracy')
    legend('moth', 'svm',  'nearest','neuralNet')
    legend('Location','southeast')
    title( [ 'Test Accuracy: ML baseline and Cyborg. ', alTag  ] )
    xlim([0, max(trList) + 5 ])
    ylim([0 100])
    grid on
    
%---------------------------------------------------------------------
    
% 2. Gain scaled by std dev of accuracy: 
figure
    hold on
    order = [3 2 4];
    for ind = 1:3
        i = order(ind);       
        plot(trList + (ind-1)*step,  ( final(:,i) - start(:,i) ) ./ ( 0.5*stdFinal(: ,i) + 0.5*stdStart(:, i)  ),...
             [colors{i} 'o'], 'markersize', 8, 'markerfacecolor', colors{i}  );
    end   

    if plotLog
        set(gca, 'XScale', 'log')
    end
    xlabel('# training samples per class')
    ylabel('gain / std dev')
    legend( 'svm',  'nearest','neuralNet')
    legend('Location','southeast')
    title( [ 'Significance: gain in accuracy, scaled by fisher discriminant' ]  ) 
    grid on
    xlim([0, max(trList) + 5 ] )
    
%----------------------------------------------------------------------
    
% 3. Std Devs of accuracy for baseline methods: 
figure
    hold on
    % moth:
    plot(trList ,  stdStart(:,1), colors{1} )
    % ML methods:
    order = [3 2 4];
    for ind = 1:3
        i = order(ind);       
        plot(trList + (ind-1)*step,  stdStart(:, i),  [colors{i} '.'], 'markersize', 24  );
    end   

    if plotLog
        set(gca, 'XScale', 'log')
    end
    xlabel('# training samples per class')
    ylabel('Std dev of Accuracy, Percentage points')
    legend( 'moth', 'svm',  'nearest','neuralNet')
    title( [ 'Std dev (in percentage points) of accuracy for baseline methods' ] ) 
    grid on
    xlim([0, max(trList) + 5 ] )
    ylim([0, 100])
    
 
%-----------------------------------------------------------------------
    
%% 4. Gains as percentages vs baseline model accuracy (scatterplot):
    
figure
    hold on
    order = [3 2 4];
    for i = order 
        plot(baselineAcc(:, i) , (gainOfMeans(:, i) ),   [colors{i} '.'], 'markersize', 24); 
        grid on
        xlabel('Baseline model accuracy (%)')
        ylabel('% gain in accuracy')
        legend(  'svm', 'nearest', 'neuralNet')
        legend('Location','northeast')
        title('Percent gain in accuracy, cyborgs vs ML baseline')
        xlim([0, 100 ] )       
        ylim([0, 100 ] )
    end
    
%-----------------------------------------------------------------------------------
    
%% 5. Relative reduction in error as percentages vs baseline model error (scatterplot):

err = 100 - baselineAcc;
     
figure
    hold on
    order = [3 2 4];
    for i = order 
        plot(baselineAcc(:, i) , (relReductionInErr(:, i) ),   [colors{i} '.'], 'markersize', 24); 
        grid on
        xlabel('Baseline model accuracy (%)')
        ylabel('% gain in accuracy')
        legend(  'svm', 'nearest', 'neuralNet')
        legend('Location','northeast')
        title('Percent reduction in error, cyborgs vs ML baseline')
        xlim([0, 100 ] )        
        ylim([0, 100 ] )
    end
    
%-----------------------------------------------------------------------------------
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
    
        
    
    
