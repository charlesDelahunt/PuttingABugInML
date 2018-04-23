
function [modelParams] = initializeConnectionMatrices_fn(modelParams)
% Generates the various connection matrices, given a modelParams struct,
% and appends them to modelParams.
% Input: 'modelParams', a struct
% Output: 'params', a struct that includes connection matrices and other model info necessary to FR evolution and plotting.

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%--------------------------------------------------------------------

% step 1: unpack input struct 'modelParams'
% step 2: build the matrices
% step 3: pack the matrices into a struct 'params' for output
% These steps are kept separate for clarity of step 2.

%% step 1: unpack modelParams:
% no editing necessary in this section.
nG = modelParams.nG;
nP = modelParams.nP;
nR = modelParams.nR;
nPI = modelParams.nPI;
nK = modelParams.nK;
nF = modelParams.nF;
nE = modelParams.nE;

tauR = modelParams.tauR;
tauP = modelParams.tauP;
tauPI = modelParams.tauPI;
tauL = modelParams.tauL;
tauK = modelParams.tauK;

cR = modelParams.cR;
cP = modelParams.cP;
cPI = modelParams.cPI;
cL = modelParams.cL;
cK = modelParams.cK;

spontRdistFlag = modelParams.spontRdistFlag;
spontRmu = modelParams.spontRmu;
spontRstd = modelParams.spontRstd;
spontRbase = modelParams.spontRbase; % for gamma dist only 

RperFFrMu = modelParams.RperFFrMu;
RperFRawNum = modelParams.RperFRawNum;
F2Rmu = modelParams.F2Rmu;
F2Rstd = modelParams.F2Rstd;
R2Gmu = modelParams.R2Gmu;
R2Gstd = modelParams.R2Gstd;
R2Pmult = modelParams.R2Pmult;  
R2Pstd = modelParams.R2Pstd;
R2PImult = modelParams.R2PImult;
R2PIstd = modelParams.R2PIstd;
R2Lmult = modelParams.R2Lmult; 
R2Lstd = modelParams.R2Lstd;

L2Gfr = modelParams.L2Gfr;
L2Gmu = modelParams.L2Gmu;
L2Gstd = modelParams.L2Gstd;

L2Rmult = modelParams.L2Rmult;  
L2Rstd = modelParams.L2Rstd;

L2Pmult = modelParams.L2Pmult; 
L2Pstd = modelParams.L2Pstd;
L2PImult = modelParams.L2PImult;
L2PIstd = modelParams.L2PIstd;
L2Lmult = modelParams.L2Lmult;
L2Lstd = modelParams.L2Lstd;

GsensMu = modelParams.GsensMu;
GsensStd = modelParams.GsensStd;

G2PImu = modelParams.G2PImu;
G2PIstd = modelParams.G2PIstd;

KperEfrMu = modelParams.KperEfrMu;
K2Emu = modelParams.K2Emu;
K2Estd = modelParams.K2Estd;

octo2Gmu = modelParams.octo2Gmu;
octo2Gstd = modelParams.octo2Gstd;
octo2Pmult = modelParams.octo2Pmult;
octo2Pstd = modelParams.octo2Pstd;
octo2PImult = modelParams.octo2PImult;
octo2PIstd = modelParams.octo2PIstd;
octo2Lmult = modelParams.octo2Lmult;
octo2Lstd = modelParams.octo2Lstd;
octo2Rmult = modelParams.octo2Rmult;
octo2Rstd = modelParams.octo2Rstd;
octo2Kmu = modelParams.octo2Kmu;
octo2Kstd = modelParams.octo2Kstd;
octo2Emu = modelParams.octo2Emu;
octo2Estd = modelParams.octo2Estd;
    
noiseR = modelParams.noiseR;
RnoiseStd = modelParams.RnoiseStd;
noiseP = modelParams.noiseP; 
PnoiseStd = modelParams.PnoiseStd;
noisePI = modelParams.noisePI;
PInoiseStd = modelParams.PInoiseStd;
noiseL = modelParams.noiseL;
LnoiseStd = modelParams.LnoiseStd;
noiseK = modelParams.noiseK;
KnoiseStd = modelParams.KnoiseStd;
noiseE = modelParams.noiseE;
EnoiseStd = modelParams.EnoiseStd;

KperPfrMu = modelParams.KperPfrMu; 
KperPIfrMu = modelParams.KperPIfrMu; 
GperPIfrMu = modelParams.GperPIfrMu; 
P2Kmu = modelParams.P2Kmu;
P2Kstd = modelParams.P2Kstd;
PI2Kmu = modelParams.PI2Kmu; 
PI2Kstd = modelParams.PI2Kstd;
kGlobalDampFactor = modelParams.kGlobalDampFactor;
kGlobalDampStd = modelParams.kGlobalDampStd;
hebMaxPK = modelParams.hebMaxPK;
hebMaxPIK = modelParams.hebMaxPIK;
hebMaxKE = modelParams.hebMaxKE;

%-------------------------------------------------------------------------

%% Step 2: Generate connection matrices
% Comment: Since there are many zero connections (ie matrices are usually
% not all-to-all) we often need to apply masks to preserve the zero connections.

% first make a binary mask S2Rbinary:
if RperFFrMu > 0
    F2Rbinary = rand(nR, nF) < RperSFrMu; % 1s and 0s.
    if makeFeaturesOrthogonalFlag 
    %     % remove any overlap in the active odors, by keeping only one non-zero entry in each row:
        b = F2Rbinary;
        for i = 1:nR 
            row = b(i,:);
            if sum(row) > 1 
                c = find(row == 1);
                t = ceil( rand(1,1)*length(c) );   % pick one index to be non-zero
                b(i,:) = 0;
                b( i, c(t) ) = 1;
            end
        end
        F2Rbinary = b;
    end
else % case: we are assigning a fixed # gloms to each S
    F2Rbinary = zeros(nR, nF);
    counts = zeros(nR,1);     % to track how many S are hitting each R
    % calc max # of S per any given glom:
    maxFperR = ceil(nF*RperFRawNum/nR);
    % connect one R to each S, then go through again to connect a 2nd R to each S, etc
    for i = 1:RperFRawNum               
        for j = 1:nF
            inds = find(counts < maxFperR);
            a = randi(length(inds));
            F2Rbinary(inds(a),j ) = 1;
            counts(inds(a)) = counts(inds(a)) + 1;
        end
    end
end

% now mask a matrix of gaussian weights:
F2R = ( F2Rmu*F2Rbinary + F2Rstd*randn(size(F2Rbinary)) ).*F2Rbinary; % the last term ensures 0s stay 0s
F2R = max(0, F2R); % to prevent any negative weights

% spontaneous FRs for Rs:
if spontRdistFlag == 1 % gaussian distribution:
    Rspont = spontRmu*ones(nG, 1) + spontRstd*randn(nG, 1);
    Rspont = max(0, Rspont);
else    % == 2 gamma distribution:
    a = spontRmu/spontRstd;
    b = spontRmu/a;  % = spontRstd
    g = makedist( 'gamma', 'a', a, 'b', b );
    Rspont = spontRbase + random(g,[nG,1]);
end

% R2G connection vector. nG x 1 col vector:
R2G  = max( 0, R2Gmu*ones(nG, 1) + R2Gstd*randn(nG, 1) ); % col vector, each entry is strength of an R in its G
                                                         % the last term prevents negative R2G effects
% now make R2P, etc, all are cols nG x 1:
R2P = ( R2Pmult + R2Pstd*randn(nG, 1) ).*R2G;
R2L = ( R2Lmult + R2Lstd*randn(nG, 1) ).*R2G;

R2PIcol = ( R2PImult + R2PIstd*randn(nG, 1) ).*R2G;
% this interim nG x 1 col vector gives the effect of each R on any PI in the R's glom.
% It will be used below with G2PI to get full effect of Rs on PIs

% Construct L2G = nG x nG matrix of lateral neurons. This is a precursor to L2P etc
L2G = max( 0, L2Gmu + L2Gstd*randn(nG) ); % kill any vals < 0
% set diagonal = 0:
L2G = L2G - diag(diag(L2G));

% are enough of these values 0?
numZero = sum(L2G(:) == 0) - nG;  % ignore the diagonal zeroes
numToKill = floor( (1-L2Gfr)*(nG^2 - nG) - numZero ); 
if numToKill > 0 % case: we need to set more vals to 0 to satisfy frLN constraint:
    L2G = L2G(:);
    randList = rand(size(L2G) ) < numToKill/(nG^2 - nG - numZero);
    L2G (L2G > 0 & randList == 1) = 0;
end
L2G = reshape(L2G,[nG,nG]);
% Structure of L2G:
% L2G(i,j) = the synaptic LN weight going to G(i) from G(j),
% ie the row gives the 'destination glom', the col gives the 'source glom'

% gloms vary widely in their sensitivity to gaba (Hong, Wilson 2014). 
% multiply the L2* vectors by Gsens + GsensStd:
gabaSens = GsensMu + GsensStd*randn(nG,1);
L2GgabaSens = L2G.*repmat(gabaSens,[1,nG]);   % ie each row is multiplied by a different value, 
                                        % since each row represents a destination glom
% this version of L2G does not encode variable sens to gaba, but is scaled by GsensMu:
L2G = L2G*GsensMu;

% now generate all the L2etc matrices:

L2R = max( 0, ( L2Rmult + L2Rstd*randn(nG) ).*L2GgabaSens );  % the last term will keep 0 entries = 0                                                 
L2P = max( 0, ( L2Pmult + L2Pstd*randn(nG) ).*L2GgabaSens );
L2L = max( 0, ( L2Lmult + L2Lstd*randn(nG) ).*L2GgabaSens );
L2PI = max( 0, ( L2Lmult + L2PIstd*randn(nG) ).*L2GgabaSens ); % Masked by G2PI later

% Ps (excitatory):
P2KconnMatrix = rand(nK, nP) < KperPfrMu; % each col is a P, and a fraction of the entries will = 1.
        % different cols (PNs) will have different numbers of 1's (~binomial dist).
P2K = max (0, P2Kmu + P2Kstd*randn(nK, nP) ); % all >= 0
P2K = P2K.*P2KconnMatrix; 
% cap P2K values at hebMaxP2K, so that hebbian training never decreases wts:
P2K = min(P2K, hebMaxPK);
% PKwt maps from the Ps to the Ks. Given firing rates P, PKwt gives the
% effect on the various Ks
% It is nK x nP with entries >= 0.

%---------------------------------------------------------------------
% PIs (inhibitory): (not used in mnist)
% 0. These are more complicated, since each PI is fed by several Gs
% 1. a) We map from Gs to PIs (binary, one G can feed multiple PI) with G2PIconn
% 1. b) We give wts to the G-> PI connections. these will be used to calc PI firing rates.
% 2. a) We map from PIs to Ks (binary), then 
% 2. b) multiply the binary map by a random matrix to get the synapse weights.

% In the moth, each PI is fed by many gloms
G2PIconn = rand(nPI, nG) < GperPIfrMu; % step 1a
G2PI = max( 0,  G2PIstd*randn(nPI, nG)  + G2PImu); % step 1b
G2PI = G2PIconn.*G2PI;  % mask with double values, step 1b (cont)
G2PI = G2PI./repmat(sum(G2PI,2), 1, size(G2PI,2) );    

% mask PI matrices: 
L2PI = G2PI*L2G;       % nPI x nG

R2PI = bsxfun(@times,G2PI, R2PIcol');  
% nG x nPI matrices, (i,j)th entry = effect from j'th object to i'th object.
% eg, the rows with non-zero entries in the j'th col of L2PI are those PIs affected by the LN from the j'th G.
% eg, the cols with non-zero entries in the i'th row of R2PI are those Rs feeding gloms that feed the i'th PI.

if nPI > 0
    PI2Kconn = rand(nK, nPI) < KperPIfrMu; % step 2a
    PI2K = max( 0, PI2Kmu + PI2Kstd*randn(nK, nPI) ); % step 2b
    PI2K = PI2K.*PI2Kconn; % mask
    PI2K = min(PI2K, hebMaxPIK);
    % 1. G2PI maps the Gs to the PIs. It is nPI x nG, doubles.
    %    The weights are used to find the net PI firing rate
    % 2. PI2K maps the PIs to the Ks. It is nK x nPI with entries >= 0.
    %    G2K = PI2K*G2PI; % binary map from G to K via PIs. not used
end
%---------------------------------------------------------------------------------------------

% K2E (excit):
K2EconnMatrix = rand(nE, nK) < KperEfrMu; % each col is a K, and a fraction of the entries will = 1.
        % different cols (KCs) will have different numbers of 1's (~binomial dist).
K2E = max (0, K2Emu + K2Estd*randn(nE, nK) ); % all >= 0
K2E = K2E.*K2EconnMatrix;  
K2E = min(K2E, hebMaxKE);
% K2E maps from the KCs to the ENs. Given firing rates KC, K2E gives the effect on the various ENs.
% It is nE x nK with entries >= 0.

% octopamine to Gs and to Ks:
octo2G = max( 0, octo2Gmu + octo2Gstd*randn(nG, 1) );  % intermediate step
% % uniform distribution (experiment):
% octo2G = max( 0, octo2Gmu + 4*octo2Gstd*rand(nG,1) - 2*octo2Gstd ); % 2*(linspace(0,1,nG) )' ); %
octo2K = max( 0, octo2Kmu + octo2Kstd*randn(nK, 1) );
% each of these is a col vector with entries >= 0

octo2P = max(0, octo2Pmult*octo2G + octo2Pstd*randn(nG, 1) ); % effect of octo on P, includes gaussian variation from P to P
octo2L = max(0, octo2Lmult*octo2G + octo2Lstd*randn(nG, 1) );
octo2R = max(0, octo2Rmult*octo2G + octo2Rstd*randn(nG, 1) );
%  % uniform distributions (experiments):
% octo2P = max(0, octo2Pmult*octo2G + 4*octo2Pstd*rand(nG, 1) - 2*octo2Pstd );
% octo2L = max(0, octo2Lmult*octo2G + 4*octo2Lstd*rand(nG, 1) - 2*octo2Lstd );
% octo2R = max(0, octo2Rmult*octo2G + 4*octo2Rstd*rand(nG, 1) - 2*octo2Rstd );
% mask and weight octo2PI:
octo2PIwts = bsxfun(@times, G2PI, octo2PImult*octo2G'); % does not include a PI-varying std term
% normalize this by taking average:
octo2PI = sum(octo2PIwts,2)./ sum(G2PIconn,2); % net, averaged effect of octo on PI. Includes varying effects of octo on Gs & varying contributions of Gs to PIs.
                                          % the 1st term = summed weights (col), 2nd term = # Gs contributing to each PI (col)
octo2E = max(0, octo2Emu + octo2Estd*randn(nE, 1) );                                          

% % each neuron has slightly different noise levels for sde use. Define noise vectors for each type:
% % Gaussian versions:
% noiseRvec = epsRstd + RnoiseSig*randn(nR, 1);
% noiseRvec = max(0, noiseRvec);   % remove negative noise entries
% noisePvec = epsPstd + PnoiseSig*randn(nP, 1);
% noisePvec = max(0, noisePvec);
% noiseLvec = epsLstd + LnoiseSig*randn(nG, 1);
% noiseLvec = max(0, noiseLvec);
noisePIvec = noisePI + PInoiseStd*randn(nPI, 1);
noisePIvec = max(0, noisePIvec);
noiseKvec = noiseK + KnoiseStd*randn(nK, 1);
noiseKvec = max(0, noiseKvec);
noiseEvec = noiseE + EnoiseStd*randn(nE, 1);
noiseEvec = max(0, noiseEvec );
% % gamma versions:
a = noiseR/RnoiseStd;
b = noiseR/a;
g = makedist( 'gamma', 'a', a, 'b', b );
noiseRvec = random(g,[nR,1]);
noiseRvec(noiseRvec > 15) = 0;   % experiment to see if just outlier noise vals boost KC noise
a = noiseP/PnoiseStd;
b = noiseP/a;
g = makedist( 'gamma', 'a', a, 'b', b );
noisePvec = random(g,[nP,1]);
noisePvec(noisePvec > 15) = 0;   % experiment to see if outlier noise vals boost KC noise
a = noiseL/LnoiseStd;
b = noiseL/a;
g = makedist( 'gamma', 'a', a, 'b', b );
noiseLvec = random(g,[nG,1]);

kGlobalDampVec = kGlobalDampFactor + kGlobalDampStd*randn(nK,1);  % each KC may be affected a bit differently by LH inhibition
%-------------------------------------------------------------------------------

%% append these matrices to 'modelParams' struct:
% no editing necessary in this section

modelParams.F2R = F2R;
modelParams.R2P = R2P;
modelParams.R2PI = R2PI;
modelParams.R2L = R2L;
modelParams.octo2R = octo2R;
modelParams.octo2P = octo2P;
modelParams.octo2PI = octo2PI;
modelParams.octo2L = octo2L;
modelParams.octo2K = octo2K;
modelParams.octo2E = octo2E;
modelParams.L2P = L2P;
modelParams.L2L = L2L;
modelParams.L2PI = L2PI;
modelParams.L2R = L2R;
modelParams.G2PI = G2PI;
modelParams.P2K = P2K;
modelParams.PI2K = PI2K;
modelParams.K2E = K2E;
modelParams.Rspont = Rspont;  % col vector

modelParams.noiseRvec = noiseRvec;
modelParams.noisePvec = noisePvec;
modelParams.noisePIvec = noisePIvec;
modelParams.noiseLvec = noiseLvec;
modelParams.noiseKvec = noiseKvec;
modelParams.noiseEvec = noiseEvec;
modelParams.kGlobalDampVec = kGlobalDampVec;


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