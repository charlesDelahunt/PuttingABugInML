
function [y] = piecewiseLinearPseudoSigmoid_fn (x, span, slope)
% Piecewise linear 'sigmoid' used for speed when squashing neural inputs in difference eqns

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%------------------------------------------------------------------

y = x*slope;
y = max(y, -span/2);
y = min(y, span/2);

