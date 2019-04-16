
function [featureArray, activePixelInds, lengthOfSide] = generateDownsampledOmniglotSet_normalizeByMaxVal_fn(preP)

% Loads the small background omniglot dataset (from Lake's website, reformatted to match structure of imageArray in 
% the MNIST version of this function). 
% Omniglot thumbnails are binary 105 x 105.
% We apply various preprocessing steps to reduce the number of pixels (each pixel will be a feature).% 
% The 'receptive field' step destroys spatial relationships, so to reconstruct a 
% (105 - 2*crop) x (105-2*crop) thumbnail (eg for viewing, or for CNN use) the active pixel indices can be embedded in a
% (105 - 2*crop)^2 x 1 col vector of zeros, then reshaped into a 12 x 12 image.%
% Modify the path for the omniglot data file as needed.
%
% Inputs: 
%   1. preP = preprocessingParams = struct with fields corresponding to relevant  variables
%
% Outputs:
%   1. featureArray = n x m x numClasses array. n = #active pixels, m = #digits
%       from each class that will be used. The 3rd dimension gives the class, 1:10   where 10 = '0'.
%   2. activePixelInds: list of pixel indices to allow re-embedding into empty thumbnail for viewing.
%   3. lengthOfSide: allows reconstruction of thumbnails given from the  feature vectors.
 
% Copyright (c) 2019 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%------------------------------------------------------------------

% Preprocessing includes:
%   1. Load omniglot background_small set.  
%   2. cropping and downsampling 
%   3. optional smearing with a gaussian (before or after downsampling)
%   3. mean-subtract, make non-negative, normalize pixel sums or by value of 90 %ile pixel
%   4. select active pixels (receptive field)

 % 1. extract omniglot:
load omniglotBackgroundSmall_imageArrayFormat.mat  % loads 4-D array 'imageArray' 
            
%   imageArray = h x w x numberImages x numberClasses 4-D array. the  classes are ordered 1 to 136

% crop, downsample, and vectorize the average images and the image stacks:

% optional pre-downsample smear with gaussian:
if isfield( preP, 'smearPreDownsample')
    if preP.smearPreDownsample
        for i = 1:size(imageArray,3)
            for j = 1:size(imageArray,4)
                this = imageArray( :, :, i, j );
                imageArray(:,:, i, j ) = imgaussfilt( this, preP.preDownsampleSmearGaussianSigma);
            end
        end
    end
end

for c = 1:size(imageArray,4)
    thisStack = imageArray(:,:,:,c);
    thisFeatureMatrix = cropDownsampleVectorizeImageStack_fn(thisStack, preP.crop, preP.downsampleRate, preP.downsampleMethod );
    featureArray(:,:,c) = thisFeatureMatrix;
end

% optional post-downsample smear with gaussian:
if isfield( preP, 'smearPostDownsample')
    if preP.smearPostDownsample
        lengthOfSide = sqrt(size(featureArray,1));
        for i = 1:size(featureArray,2)
            for j = 1:size(featureArray,3)
                this = featureArray( :,  i, j );
                thumb = reshape(this, [lengthOfSide, lengthOfSide ] );
                thumb = imgaussfilt( thumb, preP.postDownsampleSmearGaussianSigma);
                featureArray( :, i, j ) = thumb(:);
            end
        end
    end
end
clear imageArray  % to save memory

% Subtract a mean image from all feature vectors, then make values non-negative:

% a. Make an overall average feature vector, using the samples specified in 'indsToAverage'
overallAve = zeros(size(featureArray,1),1);  % initialize col vector
classAvesRaw = zeros(size(featureArray,1), size(featureArray,3));

% we will probably not average, since everything is 0, 1 so backgrounds are all 0s
if ~isempty(preP.indsToAverageGeneral)
    for c = 1:size(featureArray,3)
        classAvesRaw(:,c) = averageImageStack_fn(featureArray( : , preP.indsToAverageGeneral, c), ...
                                                                                            1 : length(preP.indsToAverageGeneral) );
        overallAve = overallAve + classAvesRaw(:,c);
    end
    overallAve = overallAve / size(featureArray,3); 

    % b. Subtract this overallAve image from all images:
    featureArray = featureArray - repmat( overallAve, [1, size(featureArray,2), size(featureArray,3) ] );
end

featureArray = max( featureArray, zeros(size(featureArray) ) ); % kill negative pixel values

% c. Normalize each image so that the 90th percentile pixel equals a fixed value:  
for i = 1:size(featureArray,2)
    for j = 1:size(featureArray,3)
        sample = featureArray(:,i,j);
        currentChosenPixelValue = prctile(sample, preP.targetedPercentile); 
        fNorm = preP.targetedPercentileValue/currentChosenPixelValue*sample;
        featureArray(:,i,j) = fNorm;
    end
end

% featureArray now consists of mean-subtracted, non-negative,
% normalized (by sum of pixels) columns, each column a vectorized thumbnail. size = numPixels x numDigitsPerClass x 10

lengthOfSide = size(featureArray,1); % save to allow sde_EM_evolution to print thumbnails.

% d. Define a Receptive Field, ie the active pixels:
% Reduce the number of features by getting rid of less-active pixels.
% If we are using an existing moth then activePixelInds is already defined, so 
% we need to load the modelParams to get the number of features (since this is defined by the AL architecture):
if preP.useExistingConnectionMatrices
    load( preP.matrixParamsFilename );    % loads 'modelParams'
    preP.numFeatures = modelParams.nF;
end
activePixelInds = selectActivePixels_fn( featureArray( :, preP.indsToCalculateReceptiveField, : ),...
                                                                                  preP.numFeatures, preP.showAverageImages );
featureArray = featureArray(activePixelInds,:,:);   % Project onto the active pixels 


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
