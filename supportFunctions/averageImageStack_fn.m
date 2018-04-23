
function [averageIm] = averageImageStack_fn(imStack, indicesToAverage)
% Average a stack of images:
% inputs:
%   1. imStack = 3-d stack (x, y, z) OR 2-d matrix (images-as-col-vecs, z)
%  Caution: Do not feed in featureArray (ie 3-d with dim 1 = feature cols, 2 = samples per class, 3 = classes)
%   2. indicesToAverage: which images in the stack to average
% Output: 
%   1. averageImage: (if input is 3-d) or column vector (if input is 2-d).

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%---------------------------------------------

% case: images are col vectors:
if length(size(imStack))  == 2
    aveIm = zeros(size(imStack,1),1);
    for i = indicesToAverage
        aveIm = aveIm + imStack(:, i);
    end
else
    aveIm = zeros(size(imStack,1), size(imStack,2));
    for i = indicesToAverage
        aveIm = aveIm + imStack(:,:,i);
    end
end    
% normalize:    
averageIm = aveIm/length(indicesToAverage);
    
