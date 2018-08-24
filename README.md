
24 August 2018

This repository contains the code used in:
1. "Putting a bug in ML: The moth olfactory network learns to read MNIST" (CB Delahunt and JN Kutz, January 2018).
2. "Insect cyborgs: Biological feature generators improve ML accuracy on limited data" (CB Delahunt and JN Kutz, August 2018).
Both papers are available on arxiv or at charlesDelahunt.github.io
The moth architecture and simulation methods are described in "Biological Mechanisms for Learning: A Computational Model of Olfactory Learning in the Manduca sexta Moth, with Applications to Neural Nets" (CB Delahunt, JA Riffell, JN Kutz, January 2018), also on arxiv or at charlesDelahunt.github.io

#--------------------------------------------------------

To train a moth:
	
	0. The main codebase simulates a moth olfactory network as it learns to read MNIST digits. It generates a model of the Manduca sexta moth olfactory network, then runs a time-stepped evolution of SDEs to train the network. The moth brain learns well from few samples (1 to 10 digits per class), with better accuracy than ML methods.

	1. 'runMothLearnerOnReducedMnist.m' runs the simulation. Key experiment parameters: goal, trPerClass, numSniffs.

	2. Run time for a simulation is 30 - 120 seconds. It is a time-stepped simulation of a fast learning system (the moth brain), not a fast system in itself.

	3. The learning task is not the typical MNIST. Rather, it is a cruder, non-spatial version of MNIST, created by downsampling and vectorizing. This is done to accommodate the size of the natural moth brain architecture (~ 60 features).

	4. 'runOtherAlgorithms' contains scripts to run Nearest-neighbor, SVM, and neural net on the downsampled MNIST. Key experiment parameters are: trPerClass, numNeighbors, boxConstraint (matlab script); trainPerClass (python script).

	5. We find that the moth brain out-performs the ML methods given in the few-samples regime (less than 20 training samples per class). Conversely, the moth brain seems to max out at around 75% accuracy. We don't know if this is due to parameter choice, or if it's an intrinsic limitation of the moth (moths can learn about 8 new odors). For example, honeybee olfactory networks are larger, though with the same basic layout. They demonstrate much richer learning behavior.

#--------------------------------------------------------

To generate insect cyborgs:

	0. The scripts in the folder "insectCyborgScripts" run experiments that use a moth brain's readout neurons as features for use by ML methods. These features substantially improve ML accuracy.

	1. In the folder 'insectCyborgScripts', run 'cyborgExperimentsMainWrapper.m'. Key experiment parameters: numRuns, trPerClassList (in 'runMothLearnerOnReducedMnistForUseInCyborg.m'.
	2. All cyborg-specific scripts and functions are in 'insectCyborgScripts', but the cyborg code calls functions in 'supportFunctions' so this must be on path.

#-------------------------------------------------------
 
Many thanks for your interest in these clever moths :) 
We hope you enjoy them! We welcome any questions, comments, (constructive) criticisms, bug reports, improvements, extensions, etc.

Charles Delahunt, delahunt@uw.edu

