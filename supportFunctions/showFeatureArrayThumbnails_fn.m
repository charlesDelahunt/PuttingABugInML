
function [] = showFeatureArrayThumbnails_fn(featureArray, numPerClass, normalize, titleString)
% Show thumbnails of inputs used in the experiment.
% Inputs: 
%   1. featureArray = either 3-D (1 = cols of features, 2 = within class samples, 3 = class)
%                               or 2-D (1 = cols of features, 2 = within class samples, no 3)
%   2. numPerClass = how many of the thumbnails from each class to show.
%   3. normalize = 1 if you want to rescale thumbs to [0 1], 0 if you don't
%   4. titleString = string

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%-------------------------------------------------

% bookkeeping: change dim if needed:
if length(size((featureArray)))  == 2 
    f = zeros(size(featureArray,1), size(featureArray,2),1);
    f(:,:,1) = featureArray;
    featureArray = f;
end

nC = size(featureArray,3);
total = nC*numPerClass;
numRows = ceil(sqrt(total/2));  % param to set
numCols = ceil(sqrt(total*2));  % param to set
vert = 1/(numRows + 1);
horiz = 1/(numCols + 1);

scrsz = get(0,'ScreenSize');
thumbs = figure('Position',[scrsz(1), scrsz(2), scrsz(3)*0.8, scrsz(4)*0.8 ]);
for class = 1:nC
    for i = 1:numPerClass
        col = numPerClass*(class-1) + i;
        thisInput = featureArray(:, i, class) ;
        % show the thumbnail of the input:        
        if normalize
            thisInput = thisInput/max(thisInput);  % renormalize, to offset effect of classMagMatrix scaling
        end
%        % reverse:
%        thisInput = (-thisInput + 1)*1.1;
        thisCol = mod( col, numCols ); 
        if thisCol == 0, thisCol = numCols; end
        thisRow = ceil( col / numCols );
        a = horiz*(thisCol - 1);
        b = 1 - vert*(thisRow);
        c = horiz;
        d = vert;
        subplot('Position', [a b c d] ), % [ left corner, bottom corner, width, height ]
        imshow(reshape(thisInput,[sqrt(length(thisInput)), sqrt(length(thisInput))] ) );   % Assumes square thumbnails
    end
   drawnow
end
% add a title at the bottom 
xlabel(titleString, 'fontweight', 'bold' ) 
drawnow


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