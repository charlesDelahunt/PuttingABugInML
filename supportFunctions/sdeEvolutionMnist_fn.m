
function [ thisRun ] = sdeEvolutionMnist_fn(tspan, initCond, time, classMagMatrix,...
    featureArray, octoHits, params, expParams, seedValue)
% To include neural noise, evolve the differential equations using euler-maruyama, milstein version
% (see Higham's Algorithmic introduction to numerical simulation of SDE)
% Called by sdeWrapper_fn.m. For use with mnist experiments.
% Inputs:
%   1. tspan: 1 x 2 vector = start and stop timepoints (sec)
%   2. initCond: n x 1 vector = starting FRs for all neurons, order-specific
%   3. time: vector of timepoints for stepping
%   4. classMagMatrix: 10 x n matrix of stimulus magnitudes. Each row  contains mags of digits from a given class
%   5. featureArray: numFeatures x numStimsPerClass x numClasses array
%   6. octoHits: 1 x length(t) vector with octopamine strengths at each timepoint
%   7. params: modelParams, including connection matrices, learning rates, etc
%   8. expParams: experiment parameters with some timing info
%   9. seedValue: for random number generation. 0 means start a new seed.
% Output:
%   thisRun: struct with fields Y (vectors of all neural timecourses as rows); T = t; 
%                 and final P2K and K2E connection matrices.

% Copyright (c) 2018 Charles B. Delahunt.  delahunt@uw.edu
% MIT License

%-------------------------------------------------

% comment: for mnist, the book-keeping differs from the odor experiment set-up.
%           Let nC = number of classes (1 - 10 for mnist).
%           The class may change with each new digit, so there is
%           be a counter that increments when stimMag changes from nonzero
%           to zero. there are nC counters.
%
% inputs:
%       1. tspan = 1 x 2 vector with start and stop times
%       2. initCond = col vector with all starting values for P, L, etc
%       3. time = start:step:stop; these are the time points for the evolution.
%          Note we assume that noise and FRs have the same step size (based on Milstein's method)
%       4. classMagMatrix = nC x N matrix where nC = # different classes (for digits, up to 10), N = length(time =
%          vector of time points). Each entry is the strength of a digit presentation.
%       5. featureArray = nF x kk x nC array, where nF = numFeatures, kk >= number of
%           puffs for that stim, and c = # classes.
%       6. octoHits = 1 x N matrix. Each entry is a strength of octopamine
%       7. params = modelParams, a struct that contains values of all connectivity matrices, noise
%            parameters, and timing params (eg when octo, stim and heb occur)
%       8. expParams = struct with timing params
%       9. seedVal = starting seed value for reproducibility. optional arg
% outputs:
%       1. T = m x 1 vector, timepoints used in evolution
%       2. Y = m x K matrix, where K contains all FRs for P, L, PI, KC, etc; and
%                  each row is the FR at a given timepoint
%
% The function uses the noise params to create a Wiener process, then evolves the FR equations with the added noise
%
% Inside the difference equations we use a piecewise linear pseudo sigmoid, rather than a true sigmoid, for speed.
%
% Note re calculating added noise:
%   We want noise to be proportional to the mean spontFR of each neuron. So
%   we need to get an estimate of this mean spont FR first. Noise is not added while neurons settle to initial SpontFR
%   values. Then noise is added, proportional to spontFR. After this  noise begins, meanSpontFRs converge to new values.
%  So there is a 'stepped' system, as follows:
%       1. no noise, neurons converge to initial meanSpontFRs = ms1
%       2. noise proportional to ms1. neurons converge to new meanSpontFRs = ms2
%       3. noise is proportional to ms2. neurons may converge to new
%          meanSpontFRs = ms3, but noise is not changed. stdSpontFRs are calculated from ms3 time period.
%   This has the following effects on simResults:
%       1. In the heat maps and time-courses this will give a period of uniform FRs.
%       2. The meanSpontFRs and stdSpontFRs are not 'settled' until after the stopSpontMean3 timepoint.
%---------------------------------------

% if argin seedValue > 0, fix the rand seed for reproducible results:
if seedValue > 0
    rng(seedValue);
end

hebStarts = expParams.hebStarts;
hebDurations = expParams.hebDurations;

hebTauPK = params.hebTauPK;
hebMaxPK = params.hebMaxPK;  % max connection weights
hebTauPIK = params.hebTauPIK; % no PIs for mnist
hebMaxPIK = params.hebMaxPIK; % no PIs for mnist
hebTauKE = params.hebTauKE;
hebMaxKE = params.hebMaxKE;

dieBackTauKE = params.dieBackTauKE ;
dieBackTauPK = params.dieBackTauPK  ;
dieBackTauPIK = params.dieBackTauPIK  ; % no PIs for mnist

sparsityTarget =  params.sparsityTarget;
octoSparsityTarget = params.octoSparsityTarget;

% unpack connection matrices from params:
F2R = params.F2R; % note S (stimuli) for odor case is replaced by F (features) for MNIST version
R2P = params.R2P;
R2PI = params.R2PI; % no PIs for mnist
R2L = params.R2L;
octo2R = params.octo2R;
octo2P = params.octo2P;
octo2PI = params.octo2PI; % no PIs for mnist
octo2L = params.octo2L;
octo2E = params.octo2E;
octoNegDiscount = params.octoNegDiscount;
L2P = params.L2P;
L2L = params.L2L;
L2PI = params.L2PI; % no PIs for mnist
L2R = params.L2R;
G2PI = params.G2PI; % no PIs for mnist
K2E = params.K2E;

P2K = params.P2K;
PI2K = params.PI2K; % no PIs for mnist
octo2K = params.octo2K;

% decay constants:
tauR = params.tauR;
tauP = params.tauP;
tauPI = params.tauPI; % no PIs for mnist
tauL = params.tauL;
tauK = params.tauK;
tauE = params.tauE;

% coefficients for sigmoids:
cR = params.cR;
cP = params.cP;
cL = params.cL;
cPI = params.cPI; % no PIs for mnist
cK = params.cK;

% numbers of objects
nC = size(classMagMatrix,1);
nF = params.nF;
nG = params.nG;
nPI = params.nPI; % no PIs for mnist
nK = params.nK;
nE = params.nE;
nP = nG;
nL = nG;
nR = nG;

% noise in individual neuron FRs. These are vectors, one vector for each type:
wRsig = params.noiseRvec;
wPsig = params.noisePvec;
wPIsig = params.noisePIvec; % no PIs for mnist
wLsig = params.noiseLvec;
wKsig = params.noiseKvec;
wEsig = params.noiseEvec;

kGlobalDampVec = params.kGlobalDampVec; % uniform 1's currently, ie LH inhibition hits all KCs equally

% steady-state RN FR, base + noise:
Rspont = params.Rspont;
RspontRatios = Rspont/mean(Rspont); % used to scale stim inputs

% param for sigmoid that squashes inputs to neurons:
slopeParam = params.slopeParam;  % slope of sigmoid at 0 = slopeParam*c/4, where c = cR, cP, cL, etc
% the slope at x = 0 = slopeParam*span/4;
kSlope = slopeParam*cK/4;
pSlope = slopeParam*cP/4;
piSlope = slopeParam*cPI/4; % no PIs for mnist
rSlope = slopeParam*cR/4;
lSlope = slopeParam*cL/4;

% end timepoints for the section used to define mean spontaneous firing rates, in order to calibrate noise.
% To let the system settle, we recalibrate noise levels to current spontaneous FRs in stages.
% This ensures that in steady state, noise levels are correct in relation to mean FRs.
startPreNoiseSpontMean1 = expParams.startPreNoiseSpontMean1;
stopPreNoiseSpontMean1 = expParams.stopPreNoiseSpontMean1;
startSpontMean2 = expParams.startSpontMean2;
stopSpontMean2 = expParams.stopSpontMean2;
startSpontMean3 = expParams.startSpontMean3;
stopSpontMean3 = expParams.stopSpontMean3;

%------------------------------------------------------------

dt = time(2) - time(1); % this is determined by start, stop and step in calling function
N = floor( (tspan(2) - tspan(1)) / dt ); % number of steps in noise evolution
T(1:N) = tspan(1):dt:tspan(2)-dt;  % the time vector

%---------------------------------------------------------

P = zeros(nP,N);
PI = zeros(nPI,N); % no PIs for mnist
L = zeros(nL,N);
R = zeros(nR, N);
K = zeros(nK, N);
E = zeros(nE, N);

% initialize the FR matrices with initial conditions:
P(:,1) = initCond( 1 : nP); % col vector
PI(:,1) = initCond( nP + 1 : nP + nPI);  % no PIs for mnist
L(:,1) = initCond( nP + nPI + 1 : nP + nPI + nL ); 
R(:,1) = initCond(nP + nPI + nL + 1: nP + nPI + nL + nR); 
K(:,1) = initCond(nP + nPI + nL + nR + 1: nP + nPI + nL + nR + nK); 
E(:,1) = initCond(end - nE + 1 : end); 
P2Kheb{1} = P2K; % '-heb' suffix is used to show that it will vary with time
PI2Kheb{1} = PI2K; % no PIs for mnist
K2Eheb{1} = K2E;
P2Kmask = P2K > 0;
PI2Kmask = PI2K > 0; % no PIs for mnist
K2Emask = K2E > 0;
newP2K = P2K; % initialize
newPI2K = PI2K; % no PIs for mnist
newK2E = K2E;

% initialize the counters for the various classes:
classCounter = zeros(size(classMagMatrix,1), 1);

% make a list of Ts for which heb is active:
hebRegion = zeros(size(T));
for i = 1:length(hebStarts)
    hebRegion(T >= hebStarts(i) & T <= hebStarts(i) + hebDurations(i) ) = 1;
end

% % DEBUG STEP:
% figure, plot(T, hebRegion), title('hebRegion vs T');

%----------------------------------------------------------

meanCalc1Done = false;  % flag to prevent redundant calcs of mean spont FRs
meanCalc2Done = false;
meanCalc3Done = false;
meanSpontR = 0*ones(size(R(:,1)));
meanSpontP = 0*ones(size(P(:,1)));
meanSpontPI = 0*ones(size(PI(:,1))); % no PIs for mnist
meanSpontL = 0*ones(size(L(:,1)));
meanSpontK = 0*ones(size(K(:,1)));
meanSpontE = 0*ones(size(E(:,1)));
ssMeanSpontP = 0*ones(size(P(:,1)));
ssStdSpontP = ones(size(P(:,1)));

maxSpontP2KtimesPval = 10; % placeholder until we have an estimate based on spontaneous PN firing rates
% The main evolution loop:
% iterate through time steps to get the full evolution:
for i = 1:N-1 % i = index of the time point
    
    step = time(2) - time(1);
    
    if T(i) < stopSpontMean3 + 5 || params.saveAllNeuralTimecourses
        oldR = R(:,i);
        oldP = P(:,i);
        oldPI = PI(:,i); % no PIs for mnist
        oldL = L(:,i);
        oldK = K(:,i);
    else % version to save memory:
        oldR = R(:,end);
        oldP = P(:,end);
        oldPI = PI(:,end);
        oldL = L(:,end);
        oldK = K(:,end);
    end
    oldE = E(:,i);
    oldT = T(i);
    
    oldP2K = newP2K; % these are inherited from the previous iteration
    oldPI2K = newPI2K; % no PIs for mnist
    oldK2E = newK2E;
    %--------------------------------------------------------
    
    % set flags to say:
    %   1. whether we are past the window where meanSpontFR is
    %       calculated, so noise should be weighted according to a first estimate of meanSpontFR (meanSpont1);
    %   2. whether we are past the window where meanSpontFR is recalculated to meanSpont2; and
    %   3. whether we are past the window where final stdSpontFR can be calculated.
    
    
    adjustNoiseFlag1 = oldT > stopPreNoiseSpontMean1;
    adjustNoiseFlag2 = oldT > stopSpontMean2;
    adjustNoiseFlag3 = oldT > stopSpontMean3;
    
    if adjustNoiseFlag1 && ~meanCalc1Done  % ie we have not yet calc'ed the noise weight vectors:
        inds = find(T > startPreNoiseSpontMean1 & T < stopPreNoiseSpontMean1);
        meanSpontP = mean(P(:,inds),2);
        meanSpontR = mean(R(:,inds),2);
        meanSpontPI = mean(PI(:,inds),2); 
        meanSpontL = mean(L(:,inds),2);
        meanSpontK = mean(K(:,inds), 2);
        meanSpontE = mean(E(:,inds), 2 );
        meanCalc1Done = 1;  % so we don't calc this again
    end
    if adjustNoiseFlag2 && ~meanCalc2Done  % ie we want to calc new noise weight vectors. This stage is surplus.
        inds = find(T > startSpontMean2 & T < stopSpontMean2);
        meanSpontP = mean(P(:,inds),2);
        meanSpontR = mean(R(:,inds),2);
        meanSpontPI = mean(PI(:,inds),2); 
        meanSpontL = mean(L(:,inds),2);
        meanSpontK = mean(K(:,inds), 2);
        meanSpontE = mean(E(:,inds), 2);
        stdSpontP = std(P(:,inds),0, 2); % for checking progress
        meanCalc2Done = 1;
    end
    if adjustNoiseFlag3 && ~meanCalc3Done % we want to calc stdSpontP for use with LH channel and maybe for use in heb:
        % maybe we should also use this for noise calcs (eg dWP). But the difference is slight.
        inds = find(T > startSpontMean3 & T < stopSpontMean3);
        ssMeanSpontP = mean(P(:,inds),2); % 'ss' means steady state
        ssStdSpontP = std(P(:,inds),0, 2);
        ssMeanSpontPI = mean(PI(:,inds),2); % no PIs for mnist
        ssStdSpontPI = std(PI(:,inds),0, 2); % no PIs for mnist
        meanCalc3Done = 1;
        % set a minimum damping on KCs based on spontaneous PN activity, 
        % sufficient to silence the MB silent absent odor:
        temp = P2K*ssMeanSpontP;
        temp = sort(temp,'ascend');
        ignoreTopN = 1;  % ie ignore this many of the highest vals
        temp = temp(1:end - ignoreTopN); % ignore the top few outlier K inputs.
        maxSpontP2KtimesPval = max(temp); % The minimum global damping on the MB.
        meanCalc3Done = 1;
    end
    
    % update classCounter:
    if i > 1
        for j = 1:nC
            if classMagMatrix(j,i-1) == 0 && classMagMatrix(j,i) > 0
                classCounter(j) = classCounter(j) + 1;
            end
        end
    end
    
    % get values of feature inputs at time index i, as a col vector.
    % This allows for simultaneous inputs by different classes, but current
    % experiments apply only one class at a time.
    thisInput = zeros(nF,1);
    thisStimClassInd = [];
    for j = 1:nC
        if classMagMatrix(j,i) > 0
            classMags = classMagMatrix(j,i) * featureArray(:,classCounter(j),j);
            thisInput = thisInput + classMags;
            thisStimClassInd = [ thisStimClassInd, j ];
        end
    end
    
    %---------------------------------------------------------------
    
    % get value at t for octopamine:
    thisOctoHit = octoHits(i); 
    % octoHits is a vector with an octopamine magnitude for each time point.
    
    %-----------------------
    % dR:
    % inputs: S = stim,  L = lateral neurons, Rspont = spontaneous FR
    % NOTE: octo does not affect Rspont. It affects R's response to input odors.
    octoMax = -L2R*oldL.*max(0,(ones(nG,1) - thisOctoHit*octo2R*octoNegDiscount));
    neur_act = (F2R*thisInput).*RspontRatios.*(ones(nG,1) + thisOctoHit*octo2R);
    Rinputs = octoMax + neur_act + Rspont;
    
    Rinputs = piecewiseLinearPseudoSigmoid_fn(Rinputs, cR, rSlope);
    
    dR = dt*(-oldR*tauR + Rinputs);
    
    %-----------------------------------------
    
    % Wiener noise:
    dWR = sqrt(dt)*wRsig.*meanSpontR.*randn(size(dR));
    % combine them:
    newR = oldR + dR + dWR;
    
    %--------------------------------------------------------
    % dP:
    Pold = -L2P*oldL.*max(0, (1 - thisOctoHit*octo2P*octoNegDiscount));
    Rold = (R2P.*oldR).*(1 + thisOctoHit*octo2P);
    Pinputs = Pold + Rold;
    % ie octo increases responsivity to positive inputs and to spont firing, and
    % decreases (to a lesser degree) responsivity to neg inputs.
    Pinputs = piecewiseLinearPseudoSigmoid_fn(Pinputs, cP, pSlope);
    
    dP = dt*( -oldP*tauP + Pinputs );
    % Wiener noise:
    dWP = sqrt(dt)*wPsig.*meanSpontP.*randn(size(dP));
    % combine them:
    newP = oldP + dP + dWP;
    
    %-----------------------------------------
    % dPI:  % no PIs for mnist
    PIinputs = max(0, (1 - thisOctoHit*octo2PI*octoNegDiscount));
    PIinputs = -L2PI*oldL.*PIinputs;
    PIinputs = PIinputs + (R2PI*oldR).*(1 + thisOctoHit*octo2PI);
    PIinputs = piecewiseLinearPseudoSigmoid_fn(PIinputs, cPI, piSlope);
    
    dPI = dt*( -oldPI*tauPI + PIinputs );
    % Wiener noise:
    dWPI = sqrt(dt)*wPIsig.*meanSpontPI.*randn(size(dPI));
    % combine them:
    newPI = oldPI + dPI + dWPI;
    
    %-----------------
    % dL:
    Linputs = max(0, (1 - thisOctoHit*octo2L*octoNegDiscount));
    Linputs = -L2L*oldL.*Linputs;
    Linputs = Linputs + (R2L.*oldR).*(1 + thisOctoHit*octo2L );
    
    
    Linputs = piecewiseLinearPseudoSigmoid_fn(Linputs, cL, lSlope);
    
    dL = dt*( -oldL*tauL + Linputs );
    % Wiener noise:
    dWL = sqrt(dt)*wLsig.*meanSpontL.*randn(size(dL));
    % combine them:
    newL = oldL + dL + dWL;
    
    %------------------------------------------------
    
    % Enforce sparsity on the KCs:
    % Global damping on KCs is controlled by sparsityTarget (during
    % octopamine, by octSparsityTarget). Assume that inputs to KCs form a
    % gaussian, and use a threshold calculated via std devs to enforce the correct sparsity.
    
    % Delays from AL -> MB and AL -> LH -> MB (~30 mSec) are ignored.
    
    % the # st devs to give the correct sparsity
    numNoOctoStds = sqrt(2)*erfinv(1 - 2*sparsityTarget);
    numOctoStds = sqrt(2)*erfinv(1 - 2*octoSparsityTarget);
    % select either octo or no-octo
    numStds = (1-thisOctoHit)*numNoOctoStds + thisOctoHit*numOctoStds; 
    % a minimum damping based on spontaneous PN activity, so that the MB is silent absent odor
    minDamperVal = 1.2*maxSpontP2KtimesPval;
    thisKinput = oldP2K*oldP - oldPI2K*oldPI; % (no PIs for mnist, only Ps)
    damper = unique(mean(thisKinput) + numStds*std(thisKinput)); 
    damper = max(damper, minDamperVal);
    
    Kinputs = oldP2K*oldP.*(1 + octo2K*thisOctoHit); % but note that octo2K == 0
    dampening = damper*kGlobalDampVec + oldPI2K*oldPI;
    pos_octo = max(0, (1 - octo2K*thisOctoHit));
    Kinputs = Kinputs - dampening.*pos_octo; % but no PIs for mnist
    
    Kinputs = piecewiseLinearPseudoSigmoid_fn(Kinputs, cK, kSlope);
    
    dK = dt*( -oldK*tauK + Kinputs );
    % Wiener noise:
    dWK = sqrt(dt)*wKsig.*meanSpontK.*randn(size(dK));
    % combine them:
    newK = oldK + dK + dWK;
    
    %----------------------------------------------------------------
    
    % readout neurons E (EN = 'extrinsic neurons'):
    % These are readouts, so there is no sigmoid.
    % octo2E == 0, since we are not stimulating ENs with octo.
    % dWE == 0 since we assume no noise in ENs.
    
    Einputs = oldK2E*oldK;    
    % (oldK2E*oldK).*(1 + thisOctoHit*octo2E); % octo2E == 0
    
    dE = dt*( -oldE*tauE + Einputs );
    % Wiener noise:
    dWE = 0; %  sqrt(dt)*wEsig.*meanSpontE.*randn(size(dE));   
    % noise = 0 => dWE == 0
    % combine them:
    newE = oldE + dE + dWE; % always non-neg
    
    %--------------------------------------------------------------------
    
    %% HEBBIAN UPDATES:
    
    % Apply Hebbian learning to P2K, K2E:
    % For ease, use 'newK' and 'oldP', 'newE' and 'oldK', ie 1 timestep of delay.
    % We restrict hebbian growth in K2E to connections into the EN of the training stimulus
    
    if hebRegion(i) % Hebbian updates are active for about half the duration of each stimulus
        
        % the PN contribution to hebbian is based on raw FR:
        tempP = oldP;
        tempPI = oldPI; % no PIs for mnist
        nonNegNewK = max(0,newK);  % since newK has not yet been made non-neg
        
        %% dP2K:
        dp2k = (1/hebTauPK) *nonNegNewK * (tempP') ;
        dp2k = dp2k.*P2Kmask; % if original synapse does not exist, it will never grow.
        
        % decay some P2K connections if wished: (not used for mnist experiments)
        if dieBackTauPK > 0
            oldP2K = oldP2K - oldP2K*(1/dieBackTauPK)*dt;
        end
        
        newP2K = oldP2K + dp2k;
        newP2K = max(0, newP2K);
        newP2K = min(newP2K, hebMaxPK*ones(size(newP2K)));
        
        %------------------------------------------------------------------------------------------------
        % dPI2K: % no PIs for mnist
        dpi2k = (1/hebTauPIK)*nonNegNewK*(tempPI');
        dpi2k = dpi2k.*PI2Kmask; % if original synapse does not exist, it will never grow.
        % kill small increases:
        temp = oldPI2K; % this detour prevents dividing by zero
        temp(temp == 0) = 1;
        keepMask = dpi2k./temp;
        keepMask = reshape(keepMask, size(dpi2k));
        dpi2k = dpi2k.*keepMask;
        if dieBackTauPIK > 0
            oldPI2K = oldPI2K - oldPI2K*(1/dieBackTauPIK)*dt;
        end
        newPI2K = oldPI2K + dpi2k;
        newPI2K = max(0, newPI2K);
        newPI2K = min(newPI2K, hebMaxPIK*ones(size(newPI2K)));
        %---------------------------------------------------------------------------------------------------
        
        %% dK2E:
        tempK = oldK;
        dk2e = (1/hebTauKE) * newE* (tempK') ; % oldK is already nonNeg
        dk2e = dk2e.*K2Emask;
        
        % restrict changes to just the i'th row of K2E, where i = ind of training stim
        restrictK2Emask = zeros(size(K2E));
        restrictK2Emask(thisStimClassInd,:) = 1;
        dk2e = dk2e.*restrictK2Emask;
        
        %---------------------------------------------------------
        
        % inactive connections for this EN die back:
        if dieBackTauKE > 0
            % restrict dieBacks to only the trained EN:
            targetMask = zeros(size(dk2e(:)));
            targetMask( dk2e(:) == 0 ) = 1;
            targetMask = reshape(targetMask, size(dk2e));
            targetMask = targetMask.*restrictK2Emask;
            dieBack = (oldK2E + 2)*(1/dieBackTauKE)*dt;
            oldK2E = oldK2E - targetMask.*dieBack; % the '+1' allows weights to die to absolute 0
        end
        
        newK2E = oldK2E + dk2e;
        newK2E = max(0,newK2E);
        newK2E = min(newK2E, hebMaxKE*ones(size(newK2E)));
        
    else % case: no heb or no octo
        newP2K = oldP2K;
        newPI2K = oldPI2K; % no PIs for mnist
        newK2E = oldK2E;
    end
    %-------------------------------------------------------------
    
    % update the evolution matrices, disallowing negative FRs. 
    if T(i) < stopSpontMean3 + 5 || params.saveAllNeuralTimecourses
        R(:,i+1) = max( 0, newR);
        P(:,i+1) = max( 0, newP);
        PI(:,i+1) = max( 0, newPI); % no PIs for mnist
        L(:,i+1) = max( 0, newL);
        K(:,i+1) = max( 0, newK);
        E(:,i+1) = newE;    
    % case: do not save AL and MB neural timecourses after the noise calibration is done, to save on memory
    else
        R = max( 0, newR);
        P = max( 0, newP);
        PI = max( 0, newPI); % no PIs for mnist
        L  = max( 0, newL);
        K  = max( 0, newK);
    end 
    
    E(:,i+1) = newE;  % always save full EN timecourses
    
end % for i = 1:N
% Time-step simulation is now over.

% combine so that each row of fn output Y is a col of [P; PI; L; R; K]:
if params.saveAllNeuralTimecourses
    Y = vertcat(P, PI, L, R, K, E);
    Y = Y';
    thisRun.Y = single(Y);  % convert to singles to save memory
else
    thisRun.Y = [];
end
 
thisRun.T = single(T');  % store T as a col
thisRun.E = single(E');  % length(T) x nE matrix
thisRun.P2Kfinal = single(oldP2K);
thisRun.K2Efinal = single(oldK2E);
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