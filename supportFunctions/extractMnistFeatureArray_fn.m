
function [imageArray ] = extractMnistFeatureArray_fn(mnist, classesToUse, imageIndices, trainOrTest)

% Extract a subset of the samples from each class, convert the images to doubles on [0 1], and
%     return a 4-D array: 1, 2 = im. 3 indexes images within a class, 4 is the class.
%
% Inputs: 
%   mnist = struct loaded by 'load mnistAll_plusSubsets'
    % with fields = training_images, test_images, training_labels, test_labels.
    % trI = mnist.train_images;
    % teI = mnist.test_images;
    % trL = mnist.train_labels;
    % teL = mnist.test_labels;
%   classesToUse = vector of the classes (digits) you want to extract
%   imageIndices = list of which images you want from each class.
%   trainOrTest = 'train' or 'test'. Determines which images you draw from
%      (since we only need a small subset, one or the other is fine).
%
% outputs:
%   imageArray = h x w x numberImages x numberClasses 4-D array
%   labels = numberClasses x numberImages matrix. Each row is a class label.

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%-------------------------------------------------

% get some dimensions:
im = mnist.trI_1(:,:,1);
[h,w] = size(im);
% initialize outputs:
imageArray = zeros(h, w, max(imageIndices), length(classesToUse) );
labels = zeros(length(classesToUse), max(imageIndices));

%--------------------------------------------------

% process each class in turn:

for c = 1:length(classesToUse)
    if strcmpi(trainOrTest, 'train') % 1 = extract train, 0 = extract test
        switch classesToUse(c)
            case 1, t = mnist.trI_1;
            case 2, t = mnist.trI_2;
            case 3, t = mnist.trI_3;
            case 4, t = mnist.trI_4;
            case 5, t = mnist.trI_5;
            case 6, t = mnist.trI_6;
            case 7, t = mnist.trI_7;
            case 8, t = mnist.trI_8;
            case 9, t = mnist.trI_9;
            case 10, t = mnist.trI_0;
        end
    else
        switch classesToUse(c)
            case 1, t = mnist.teI_1;
            case 2, t = mnist.teI_2;
            case 3, t = mnist.teI_3;
            case 4, t = mnist.teI_4;
            case 5, t = mnist.teI_5;
            case 6, t = mnist.teI_6;
            case 7, t = mnist.teI_7;
            case 8, t = mnist.teI_8;
            case 9, t = mnist.teI_9;
            case 10, t = mnist.teI_0;
        end
    end
    % now we have the correct image stack: 
    % convert to double:
    t = double(t)/256;
    imageArray(:,:,imageIndices, c) = t(:,:,imageIndices);
end

% get rid of any leading zero images caused by skipping the first N:
imageArray = imageArray(:,:,imageIndices,:);
%---------------------------------------------------------------------------


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