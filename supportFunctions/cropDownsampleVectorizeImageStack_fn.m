
function [imColArray] = cropDownsampleVectorizeImageStack_fn(imStack, cropVal, downsampleVal, downsampleMethod )

% For each image in a stack of images: Crop, then downsample, then make into a col vector. 
% Inputs:
%   1. imStack = width x height x numImages array
%   2. cropVal = number of pixels to shave off each side. can be a scalar or a
%       4 x 1 vector: top, bottom, left, right.
%   3. downsampleVal = amount to downsample
%   4. downsampleMethod: if 0, do downsampling by summing square patches. If 1, use bicubic interpolation.
% Output: 
%   1. imArray = a x numImages matrix, where a = number of pixels in the cropped and downsampled images

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%------------------------------------------------------------------

if length(cropVal) == 1  
    cropVal = cropVal*ones(1,4);
end

if length(size(imStack) ) == 3
    [h, w, z] = size(imStack);
else
    [h, w ] = size(imStack);
    z = 1;
end

width = cropVal(3) + 1:w - cropVal(4);
height = cropVal(1) + 1:h - cropVal(2);

 % crop, downsample, vectorize the thumbnails one-by-one:
for s = 1:z
    t = imStack(:,:, s);
    t = t(height, width);
    d = downsampleVal;
    % to downsample, do bicubic interp or sum the blocks:
    if downsampleMethod %  ie == 1 (or any non-zero num): bicubic
        t2 = imresize(t, 1 / downsampleVal);
    else   % downsampleMethod == 0: sum 2 x 2 blocks
        for i = 1:length(height) / d
            for j = 1:length(width) / d
                b = t( (i-1)*d + 1:i*d, (j-1)*d + 1:j*d );
                t2(i,j) = sum(b(:));
            end
        end
    end
    
    t2 = t2/max(t2(:));        % normalize to [0,1]
    imColArray(:,s) = t2(:);   
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