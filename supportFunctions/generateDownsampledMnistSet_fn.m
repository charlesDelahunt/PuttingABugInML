
function [featureArray, activePixelInds, lengthOfSide] = generateDownsampledMnistSet_fn(preP)

% Loads the mnist dataset (the version in PMTK3 by Kevin Murphy, Machine Learning 2012), 
% then applies various preprocessing steps to reduce the number of pixels (each pixel will be a feature).% 
% The 'receptive field' step destroys spatial relationships, so to reconstruct a 
% 12 x 12 thumbnail (eg for viewing, or for CNN use) the active pixel indices can be embedded in a
% 144 x 1 col vector of zeros, then reshaped into a 12 x 12 image.%
% Modify the path for the mnist data file as needed.
%
% Inputs: 
%   1. preP = preprocessingParams = struct with fields corresponding to relevant  variables
%
% Outputs:
%   1. featureArray = n x m x 10 array. n = #active pixels, m = #digits
%       from each class that will be used. The 3rd dimension gives the class, 1:10   where 10 = '0'.
%   2. activePixelInds: list of pixel indices to allow re-embedding into empty thumbnail for viewing.
%   3. lengthOfSide: allows reconstruction of thumbnails given from the  feature vectors.
 
% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%------------------------------------------------------------------

% Preprocessing includes:
%   1. Load MNIST set.  
%   2. cropping and downsampling 
%   3. mean-subtract, make non-negative, normalize pixel sums
%   4. select active pixels (receptive field)

 % 1. extract mnist:
load mnistAll/mnistAll_plusSubsets.mat  % loads struct 'mnist' with fields = 
             % .training_images, .test_images, .training_labels, .test_labels (ie the original data from PMTK3)
             % AND parsed by class. These fields are used to assemble the imageArray:
             % .trI_* = train_images of class *; 
             % .teI_* = test_images of class *; 
             % .trL_* = train_labels of class *; 
             % .teL_* = test_labels of class *;

% extract the required images and classes. 
imageIndices = 1 : preP.maxInd;
[imageArray] = extractMnistFeatureArray_fn( mnist, preP.classLabels, imageIndices, 'train');
%   imageArray = h x w x numberImages x numberClasses 4-D array. the classes are ordered 1 to 10 (10 = '0')

% crop, downsample, and vectorize the average images and the image stacks:
for c = 1:size(imageArray,4)
    thisStack = imageArray(:,:,:,c);
    thisFeatureMatrix = cropDownsampleVectorizeImageStack_fn(thisStack, preP.crop, preP.downsampleRate, preP.downsampleMethod );
    featureArray(:,:,c) = thisFeatureMatrix;
end

clear imageArray  % to save memory

% Subtract a mean image from all feature vectors, then make values non-negative:

% a. Make an overall average feature vector, using the samples specified in 'indsToAverage'
overallAve = zeros(size(featureArray,1),1);  % initialize col vector
classAvesRaw = zeros(size(featureArray,1), size(featureArray,3));
for c = 1:size(featureArray,3)
    classAvesRaw(:,c) = averageImageStack_fn(featureArray( : , preP.indsToAverageGeneral, c), ...
                                                                                        1 : length(preP.indsToAverageGeneral) );
    overallAve = overallAve + classAvesRaw(:,c);
end
overallAve = overallAve / size(featureArray,3); 

% b. Subtract this overallAve image from all images:
featureArray = featureArray - repmat( overallAve, [1, size(featureArray,2), size(featureArray,3) ] );
featureArray = max( featureArray, 0 ); % kill negative pixel values

% c. Normalize each image so the pixels sum to the same amount:         
fSums = sum(featureArray,1);
fNorm = preP.pixelSum*featureArray./repmat(fSums, [size(featureArray,1), 1, 1 ] );
featureArray = fNorm;
% featureArray now consists of mean-subtracted, non-negative,
% normalized (by sum of pixels) columns, each column a vectorized thumbnail. size = 144 x numDigitsPerClass x 10

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
