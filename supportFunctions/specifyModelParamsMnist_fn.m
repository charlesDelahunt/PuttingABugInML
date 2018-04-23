
function [modelParams] = specifyModelParamsMnist_fn( nF, goal )

% [modelParams] = specifyModelParamsMnist_fn(modifications ) generates the param structure for the moth, ie 
%   the template that is used to populate connection matrices and to control behavior.
% Input:
%   1. nF = number of features. This determines the number of neurons in each layer.
%   2. goal = measure of learning rate: goal = N means we expect the moth to hit max accuracy when
%                  trained on N samples per class. So goal = 1 gives a fast learner, goal = 20 gives a slower learner.
% Output:
%   1. modelParams = struct ready to pass to 'initializeConnectionMatrices_fn'

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%-------------------------------------------------

% The following abbreviations are used:
% n* = number of *
% G = glomerulus (so eg nG = number of glomeruli)
% R = response neuron (from antennae): this concept is not used. We
%           use the stim -> glomeruli connections directly
% P = excitatory projection neuron. note sometimes P's stand in for gloms
%       in indexing, since they are 1 to 1
% PI = inhibitory projection neuron
% L = lateral neuron (inhibitory)
% K = kenyon cell (in MB)
% F = feature   (this is a change from the original moth/odor regime, where
% each stim/odor was identified with its own single feature).
% S (not used in this function) = stimulus class.
% fr = fraction
% mu = mean
% std = standard deviation
% _2_ = synapse connection to, eg P2Kmu = mean synapse strength from PN to KC
% octo = octopamine delivery neuron
%

% General structure of synaptic strength matrices:
% rows give the 'from' a synapse
% cols give the 'to'. 
% so M(i,j) is the strength from obj(i) to obj(j)

% below, 'G' stands for glomerulus. Glomeruli are not explicitly part of the equations, but
% the matrices for LN interconnections, PNs, and RNs are indexed according to the G

if nargin < 2, goal = 20; end 
if nargin < 1, nF = 85; end

nG = nF; 
nP = nG; % Pn = # excitatory Pn. one per glomerulus
nR = nG; % RNs, one per glom
% for now assume no pheromone gloms. Can add later, along with special Ps, PIs, and Ls
nK2nGRatio = 30;
nK = nK2nGRatio*nG;   % number of kenyon cells (in MB)
% get count of inhibitory projection neurons = PIs:
PIfr = 0.05; %  for mnist experiments there are no inhibitory projection neurons (PIs). But make a couple placeholders.
nPI = round(nG*PIfr);  % these are in addition to nP
% note that outputs P and PI only affect KCs
nE = 10;    % extrinsic neurons in eg beta-lobe of MB, ie the read-out/decision neurons

%------------------------------------------------

%% Hebbian learning rates:

% For K2E ie MB -> EN. Very important. Most of the de facto plasticity is in K2E:
hebTauKE = 0.02*goal; % controls learning rate for K2E weights. 1/decay rate. Higher means slower decay.
dieBackTauKE = 0.5*goal; % 1/decay rate. Higher means slower decay. 
% dieBackTauKE and hebTauKE want to be in balance.

% For P2K ie AL -> MB:
hebTauPK =  5e3*goal;  % learning rate for P2K weights. Higher means slower. 
% Very high hebTauPK means that P2K connections are essentially fixed.
% Decay: There is no decay for P2K weights:
dieBackTauPK = 0; %  If > 0, divide this fraction of gains evenly among all nonzero weights, and subtract.
dieBackTauPIK = 0; % no PIs in mnist moths

%------------------------------------------------------------------------
%% Time constants for the diff eqns:
% An important param: the same param set will give different results if only this one is changed.
% assume 3*(1/time constant) = approx 1+ seconds (3 t.c -> 95% decay)
tau = 7; 
tauR = tau;
tauP = tau;
tauPI = tau;
tauL = tau;
tauK = tau;
tauE = tau;

% stdmoid range parameter for the diff eqns:
C = 10.5; 
cR = C; 
cP = C;
cPI = C;
cL = C;
cK = C; 

% stdmoid slope param for the diff eqns:
% slope of stdmoid at zero = C*slopeParam/4
desiredSlope = 1;  
slopeParam = desiredSlope*4/C;   % a convenience variable. 'desiredSlope' is the one to adjust.  

% stdmoid param to make range 0:C or -C/2:C/2
symmetricAboutZero = 1;  %  '0' means: [-inf:inf] -> [0, C], 0 -> C/2. '1' means: [-inf:inf] -> [-c/2,c/2], 0 -> 0. 
stdFactor= 0.1;   % this value multiplies the mean value of a connection matrix to give the STD.
% It applies to all connection matrices, ie it is a global parameter.

%% Parameters to generate connection matrices between various types of neurons:

% Typically the effect of an input to G (ie a glomerulus), as
% passed on to P, L, PI, and R within the glomerulus, is set = mult*effectOnG + std*NormalDist. 
% 'mult' is different for P, L, and R. That is, a stdnal entering the
% glomerulus affects the different neuron types differently.

% mu and std define mean and std of connection matrix entries:

% KEY POINT: R2L, R2P, and R2PI should be highly correlated, since all of them
% depend on the R2G connectivity strength. For this reason, use an
% intermediate connectivity matrix R2G to give the base (glomerular) levels of
% connectivity. Then derive R2P, R2L from R2G. Similarly, define L2G
% first, then derive L2L, L2P, L2R).

% For mnist, each pixel is a feature, and goes to exactly one receptor
% neuron RN, ie to exactly one glomerulus G:
RperFFrMu = 0;  %  used in odor simulations, disabled for mnist.
RperFRawNum = 1;  % ie one RN (equiv one Glom) for each F (feature, ie pixel).

F2Rmu = 100 / nG;  % controls how strongly a given feature will affect G's
F2Rstd = 0;  

% ------------------------------------------------------------------

% R characteristics:
R2Gmu = 5;  % controls how strongly R's affect their G's
R2Gstd = 0;  

% used to create R2P, using as base R2G:
R2Pmult =  4;  % this multiplies R2G values, to give R2P values
R2Pstd = stdFactor*R2Pmult;  % variation in R2P values, beyond the conversion from R2G values

R2PImult = 0;  %  no PIs for mnist
R2PIstd = 0; 

R2Lmult =  0.75;  % so R has much weaker effect on LNs than on PNs (0.75 vs 4)
R2Lstd = stdFactor*R2Lmult; 

%-----------------------------------------------------------------------

% define spontaneous steady-state firing rates (FRs) of RNs:
spontRdistFlag = 2; % 1 = gaussian, 2 = gamma + base.
spontRmu = 0.1;  
spontRstd = 0.05*spontRmu; 
spontRbase = 0; % for gamma only
%  if using gamma, params are derived from spontRmu and spontRstd.
    
%-----------------------------------------------------------------------

% L2* connection matrices.

% KEY POINT: L2R, L2L, L2P, and L2PI should be highly correlated, since all
% depend on the L2G connectivity strength. For this reason, use an
% intermediate connectivity matrix L2G to give the base levels of
% connectivity. Then derive L2P, L2R, L2L from L2G.

% Define the distribution of LNs, ie inhib glom-glom synaptic strengths:
L2Gfr =  0.8;  % the fraction of possible LN connections (glom to glom) that are non-zero.   % hong: fr = most or all
L2Gmu = 5;                                                                                   
L2Gstd = stdFactor*L2Gmu;  

% sensitivity of G to gaba, ie sens to L, varies substantially with G. 
% Hong-Wilson says PNs (also Gs) sensitivity to gaba varies as N(0.4,0.2). Note this distribution is the end-result of 
% many interactions, not a direct application of GsensMu and Gsensstd.
GsensMu = 0.6;  
GsensStd = 0.2*GsensMu;  
% Note: Gsensstd expresses variation in L effect, so if we assume that P, L, and R are all similarly gaba-resistent
% within a given G, set L2Pstd etc below = 0.

%---------------------------------------------------

% define effect of LNs on various components:

% used to create L2R, using as base L2G:
L2Rmult = 6/ (nG*L2Gfr); 
L2Rstd =  0.1*L2Rmult;                                                                       

% used to create L2P, using as base L2G:
L2Pmult = 2/ (nG*L2Gfr);  
L2Pstd = stdFactor*L2Pmult;  % variation in L2P values, beyond the conversion from L2G values             
% used to create L2PI, using as base L2G:
L2PImult =  L2Pmult;
L2PIstd =  stdFactor*L2PImult; 
% used to create L2L, using as base L2G:
L2Lmult = 2/ (nG*L2Gfr);  
L2Lstd =  stdFactor*L2Lmult; 

% weights of G's contributions to PIs, so different Gs will contribute differently to net PI FR:
G2PImu = 1; % 0.6;  
G2PIstd = 0; %   0.1*G2PImu; 

%------------------------------------------------------------

% Sparsity in the KCs:
% KCs are globally damped by the LH (or perhaps by a neuron within the MB). The
% key net effect is sparsity in the KCs.
% For mnist experiments, the KC sparsity level is directly controlled by two parameters.

sparsityTarget =  0.05;  
octoSparsityTarget = 0.075;  %  used when octopamine is active 

kGlobalDampFactor = 1;  % used to make a vector to deliver damping to KCs
kGlobalDampStd = 0; % allows variation of damping effect by KC. 
% Effect of the above is to make global damping uniform on KCs.

%---------------------------------------------------------------------

% AL to MB, ie PN to KC connection matrices:

% Define distributions to describe how Ps connect to Ks, ie synapses:

% a) excitatory PNs. We need # of KCs connected to each PN:
% first give the # Ps that feed each K
numPperK = 10; 
KperPfrMu = numPperK / nG;   % the mean fraction of KCs a given 'PN' connects to. 
                  % Note that here 'PN' = glom. That is, multiple PNs coming 
                  % out of a single glom are treated as one PN. 
                  % assuming 5 true PNs per glom, and 2000 KCs, 0.2 means: 
                  % each glom -> 0.2*2000 = 400 KC, so each true PN -> 80
                  % KCs, so there are 300*80 PN:KC links = 24000 links, so
                  % each KC is linked to 24k/2k = 12 true PNs (see Turner 2008).
                  % The actual # will vary as a binomial distribution
                  % simpler calc: KperPfrMu*nP = "true" PNs per K.
                  % The end result is a relatively sparse P2K connection
                  % matrix with very many zeros. The zeros are permanent (not
                  % modifiable by plasticity).
                  
% b) inhibitory PNs (not used in mnist experiments, but use placeholders to avoid crashes. Weights are 0). 
%     We need the # of Gs feeding into each PI and # of Ks the PI goes to:
KperPIfrMu =    1/nK; % KperPfrMu; the ave fraction of KCs a PI connects to 
                                                     % The actual # will vary as a binomial distribution
GperPIfrMu = 5/nG;  % ave fraction of Gs feeding into a PI (for Ps, there is always 1 G) 
                  % The actual # will vary as a binomial distribution
                  % real moth: about 3 - 8 which corresponds to GperPIfrMu = 0.1

% Define distribution that describes the strength of PN->KC synapses:
% these params apply to both excitatory and inhibitory PNs. These are plastic, via
% PN stimulation during octo stim.
pMeanEstimate = R2Gmu*R2Pmult*(spontRbase + spontRmu);  % ~2 hopefully. For use in calibrating P2K
piMeanEstimate = pMeanEstimate*KperPfrMu/KperPIfrMu;   % not used in mnist

P2Kmultiplier = 44;     

P2Kmu = P2Kmultiplier / (numPperK*pMeanEstimate); 
P2Kstd = 0.5*P2Kmu;    % bigger variance than most connection matrices

PI2Kmu = 0; 
PI2Kstd = 0; 

hebMaxPK = P2Kmu + 3*P2Kstd; % ceiling for P2K weights
hebTauPIK = hebTauPK;  % irrelevant since PI2K weights == 0
hebMaxPIK = PI2Kmu + 3*PI2Kstd;
%---------------------------------------------------------------

% KC -> EN connections:
% Start with all weights uniform and full connectivity. Training rapidly individuates the connectivities of ENs.
KperEfrMu = 1;  % what fraction of KCs attach to a given EN
K2Emu =  3;     % strength of connection
K2Estd = 0;     % variation in connection strengths
hebMaxKE = 20*K2Emu + 3*K2Estd;   % max allowed connection weights (min = 0)

%---------------------------------------------------------------

% distribution of octopamine -> glom strengths (small variation):
% NOTES:
%   1. these values assume octoMag = 1
%   2. if using the multiplicative effect, we multiply the inputs to neuron N by (1 +
%      octo2N) for positive inputs, and (1-octo2N) for negative inputs. So we
%      probably want octo2N to be close to 0. It is always non-neg by
%      construction in 'initializeConnectionMatrices.m'

% In the context of dynamics eqns, octopamine reduces a neuron's responsivity to inhibitory inputs, but 
% it might do this less strongly than it increases the neuron's responsivity to excitatory inputs:
octoNegDiscount = 0.5; % < 1 means less strong effect on neg inputs. 

% since octo strengths are correlated with a glom, use same method as for gaba sensitivity:
octo2Gmu =  1.5;  
octoStdFactor= 0;      % ie make octo effect on all gloms the same
octo2Gstd = octoStdFactor*octo2Gmu; 

% Per jeff (may 2016), octo affects R, and may also affect P, PI, and L. First try "octo affects R only", then add the others. 
% used to create octo2R, using as base octo2G.
octo2Rmult = 2;  
octo2Rstd = octoStdFactor*octo2Rmult;  
% used to create octo2P, using as base octo2G:
octo2Pmult = 0;   % octo2P = 0 means we only want to stimulate the response to actual odor inputs (ie at R):
%  octo2Pmult > 0 mean we stimulate P's responsiveness to all signals, so we amplify noise as well.
octo2Pstd = octoStdFactor*octo2Pmult;
% used to create octo2PI, using as base octo2G:  unused for mnist experiments.
octo2PImult = 0;  
octo2PIstd = octo2Pstd; 

% used to create octo2L, using as base octo2G:
octo2Lmult = 1.6;    % This must be > 0 if Rspont is not affected by octo in sde_dynamics (ie if octo only affects 
                     % RN's reaction to odor), since jeff's data shows that some Rspont decrease with octo                              
octo2Lstd =  octoStdFactor*octo2Lmult;  

% end of AL-specific octopamine param specs.

% Distribution of octopamine -> KC strengths (small variation):
% We assume no octopamine stimulation of KCs, ie higher KC activity follows
% from higher AL activity due to octopamine, not from direct octopamine effects on KCs.
octo2Kmu = 0;   
octo2Kstd = octoStdFactor*octo2Kmu;

octo2Emu = 0;  % for completeness, not used.
octo2Estd = 0; 

%------------------------------------------------------------------------

% Noise parameters for noisy AL neurons.
% Define distributions for random variation in P,R,L and K vectors at each step of the simulation:
% These control random variations in the various neurons that are applied at
% each time step as 'epsG' and 'epsK'. This might serve to break up potential oscillations.
noise = 1;  
noiseStdFactor= 0.3;  

noiseR = noise;
RnoiseStd = noiseStdFactor*noiseR;  
noiseP = noise; 
PnoiseStd = noiseStdFactor*noiseP;
noiseL = noise;
LnoiseStd = noiseStdFactor*noiseL;
noisePI = noise;
PInoiseStd = noiseStdFactor*noisePI;
noiseK = 0; 
KnoiseStd = noiseStdFactor*noiseK;
noiseE = 0;
EnoiseStd = 0;

%-------------------------------------------------------------------------------------------

% save everything into a struct for export. 
% Lines below here should not be modified except when parameters are added or removed.

modelParams.nF = nF;
modelParams.nG = nG;
modelParams.nR = nR;
modelParams.nP = nP;
modelParams.nPI = nPI;
modelParams.nK = nK;
modelParams.nE = nE;
modelParams.nK2nGRatio = nK2nGRatio;

modelParams.tauR = tauR;
modelParams.tauP = tauP;
modelParams.tauPI = tauPI;
modelParams.tauL = tauL;
modelParams.tauK = tauK;
modelParams.tauE = tauE;

modelParams.cR = cR;
modelParams.cP = cP;
modelParams.cPI = cPI;
modelParams.cL = cL;
modelParams.cK = cK;

modelParams.slopeParam = slopeParam; 
modelParams.symmetricAboutZero = symmetricAboutZero;

modelParams.spontRmu = spontRmu; 
modelParams.spontRstd = spontRstd;
modelParams.spontRbase = spontRbase;
modelParams.spontRdistFlag = spontRdistFlag;

modelParams.RperFFrMu = RperFFrMu;
modelParams.RperFRawNum = RperFRawNum;
modelParams.F2Rmu = F2Rmu;
modelParams.F2Rstd = F2Rstd; 

modelParams.R2Gmu = R2Gmu;
modelParams.R2Gstd = R2Gstd;
modelParams.R2Pmult = R2Pmult;  
modelParams.R2Pstd = R2Pstd; 
modelParams.R2PImult = R2PImult;
modelParams.R2PIstd = R2PIstd;
modelParams.R2Lmult = R2Lmult; 
modelParams.R2Lstd = R2Lstd;

modelParams.L2Gfr = L2Gfr; 
modelParams.L2Gmu = L2Gmu;
modelParams.L2Gstd = L2Gstd; 

modelParams.L2Rmult = L2Rmult;   
modelParams.L2Rstd = L2Rstd;

modelParams.L2Pmult = L2Pmult; 
modelParams.L2Pstd = L2Pstd;
modelParams.L2PImult = L2PImult;
modelParams.L2PIstd = L2PIstd;
modelParams.L2Lmult = L2Lmult;
modelParams.L2Lstd = L2Lstd;

modelParams.GsensMu = GsensMu;
modelParams.GsensStd = GsensStd;

modelParams.PIfr = PIfr;
modelParams.G2PImu = G2PImu;
modelParams.G2PIstd = G2PIstd;

modelParams.KperPfrMu = KperPfrMu; 
modelParams.KperPIfrMu = KperPIfrMu; 
modelParams.GperPIfrMu = GperPIfrMu; 
modelParams.P2Kmu= P2Kmu;
modelParams.P2Kstd = P2Kstd;
modelParams.PI2Kmu = PI2Kmu; 
modelParams.PI2Kstd = PI2Kstd;

modelParams.kGlobalDampFactor = kGlobalDampFactor;
modelParams.kGlobalDampStd = kGlobalDampStd;

modelParams.KperEfrMu = KperEfrMu;
modelParams.K2Emu = K2Emu;
modelParams.K2Estd = K2Estd;

modelParams.octoNegDiscount = octoNegDiscount;
modelParams.octostdFactor= octoStdFactor;
modelParams.octo2Gmu = octo2Gmu;
modelParams.octo2Gstd = octo2Gstd;
modelParams.octo2Pmult = octo2Pmult;
modelParams.octo2Pstd = octo2Pstd;
modelParams.octo2PImult = octo2PImult;
modelParams.octo2PIstd = octo2PIstd;
modelParams.octo2Lmult = octo2Lmult;
modelParams.octo2Lstd = octo2Lstd;
modelParams.octo2Rmult = octo2Rmult;
modelParams.octo2Rstd = octo2Rstd;
modelParams.octo2Kmu = octo2Kmu;
modelParams.octo2Kstd = octo2Kstd;
modelParams.octo2Emu = octo2Emu;
modelParams.octo2Estd = octo2Estd;

modelParams.hebTauPK = hebTauPK;
modelParams.hebTauPIK = hebTauPIK;
modelParams.hebTauKE = hebTauKE;
modelParams.hebMaxPK = hebMaxPK;
modelParams.hebMaxPIK = hebMaxPIK;
modelParams.hebMaxKE = hebMaxKE;

modelParams.sparsityTarget = sparsityTarget;
modelParams.octoSparsityTarget = octoSparsityTarget;

modelParams.dieBackTauPK =  dieBackTauPK;
modelParams.dieBackTauPIK =  dieBackTauPIK;
modelParams.dieBackTauKE  =  dieBackTauKE;

modelParams.noiseR = noiseR;
modelParams.RnoiseStd = RnoiseStd;
modelParams.noiseP = noiseP;
modelParams.PnoiseStd = PnoiseStd;  
modelParams.noisePI = noisePI;
modelParams.PInoiseStd = PInoiseStd;
modelParams.noiseL = noiseL;
modelParams.LnoiseStd = LnoiseStd;
modelParams.noiseK = noiseK;
modelParams.KnoiseStd = KnoiseStd;
modelParams.noiseE = noiseE;
modelParams.EnoiseStd = EnoiseStd;


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