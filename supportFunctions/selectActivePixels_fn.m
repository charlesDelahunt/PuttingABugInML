
function [activePixelInds] = selectActivePixels_fn(featureArray, numFeatures, showImages)
% Select the most active pixels, considering all class average images, to use as features.
% Inputs:
%    1. featureArray: 3-D array nF x nS x nC, where nF = # of features, nS = # samples per class, nC = number of classes.
%        As created by generateDwnsampledMnistSet_fn.m
%    2. numFeatures: The number of active pixels to use (these form the receptive field).
%    3. showImages:  1 means show average class images, 0 = don't show.
% Output:
%   1. activePixelInds: 1 x nF vector of indices to use as features.
%       Indices are relative to the vectorized thumbnails (so between 1 and 144).

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%-----------------------------------------------------------------------------------------

% make a classAves matrix, each col a class ave 1 to 10 (ie 0), and add a col for the overallAve
numPerClass = size(featureArray,2);
cA = zeros(size(featureArray,1), size(featureArray,3) + 1); 

for i = 1:size(featureArray,3)
    % change dim of argin 1 to 'averageImageStack'
    temp = zeros(size(featureArray,1), size(featureArray,2));
    temp(:,:) = featureArray(:,:,i);
    cA(:,i) = averageImageStack_fn(temp, 1:numPerClass );
end
% last col = overall average image: 
cA(:,end) = sum( cA(:,1:end-1), 2) / (size(cA,2) - 1) ;

% normed version. Do not rescale the overall average:
z = max(cA);
caNormed = cA./repmat( [z(1:end-1), 1], [size(cA,1),1]);
num = size(caNormed,2);

%-----------------------------------------------------------------------------    

% select most active 'numFeatures' pixels:
this = cA( : , 1:end - 1 );
thisLogical = zeros( size( this ) );
vals = sort( this(:), 'descend' );    % all the pixel values from all the class averages, in descending order  
% start selecting the highest-valued pixels:
stop = 0;
while ~stop
    thresh = max(vals);
    thisLogical( this >= thresh ) = 1;
    activePixels = sum( thisLogical,  2 );  % sum the rows. If a class ave had the i'th pixel, selected, keptPixels(i) > 0
    stop = sum(activePixels > 0) >= numFeatures;  % we have enough pixels.
    vals = vals(vals < thresh);  % peel off the value(s) just used.
end
activePixelInds = find( activePixels > 0 );
    
%------------------------------------------------------------------------

if showImages
    % plot the normalized classAves pre-ablation:
    normalize = 0;
    titleStr = 'class aves, all pixels';
    showFeatureArrayThumbnails_fn(caNormed, size(caNormed,2), normalize, titleStr)
 
    % look at active pixels of the classAves, ie post-ablation:
    normalize = 0;
    caActiveOnly = zeros(size(caNormed));
    caActiveOnly(activePixelInds, : ) = caNormed(activePixelInds, :) ; 
    titleStr = 'class aves, active pixels only';
    showFeatureArrayThumbnails_fn(caActiveOnly, size(caActiveOnly,2), normalize, titleStr)
    
end 


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