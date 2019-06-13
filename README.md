
12 June 2019

This repository contains the code used in:
1. "Putting a bug in ML: The moth olfactory network learns to read MNIST" (CB Delahunt and JN Kutz), Neural Networks 2019, https://doi.org/10.1016/j.neunet.2019.05.012
2. "Insect cyborgs: Bio-mimetic feature generators improve ML accuracy on limited data" (CB Delahunt and JN Kutz), 2018, https://arxiv.org/abs/1808.08124

The moth architecture and simulation methods are described in "Biological Mechanisms for Learning: A Computational Model of Olfactory Learning in the Manduca sexta Moth, with Applications to Neural Nets" (CB Delahunt, JA Riffell, JN Kutz), Frontiers in Neuroscience 2018, https://doi.org/10.3389/fncom.2018.00102

All papers are also available at charlesDelahunt.github.io

Matlab version: This repo, instructions below.
Python version: (thanks to Adam P. Jones) https://github.com/meccaLeccaHi/pyMoth

#--------------------------------------------------------

To train a moth:
	
	0. The main codebase simulates a moth olfactory network as it learns to read MNIST digits. It generates a model of the Manduca sexta moth olfactory network, then runs a time-stepped evolution of SDEs to train the network. The moth brain learns well from few samples (1 to 10 digits per class), with better accuracy than ML methods.

	1. 'runMothLearnerOnReducedMnist.m' runs the simulation. Key experiment parameters: goal, trPerClass, numSniffs.

	2. Run time for a simulation is 30 - 120 seconds. It is a time-stepped simulation of learning in the moth brain.

	3. The learning task is not the typical MNIST. Rather, it is a cruder, non-spatial version of MNIST, created by downsampling and vectorizing. This is done to match the size of the natural moth brain architecture (~ 60 features).

	4. 'runOtherAlgorithms' contains scripts to run Nearest-Neighbor, SVM, and Neural Net on the downsampled MNIST. 

	5. We find that the moth brain out-performs the ML methods given in the few-samples regime (less than 20 training samples per class). Conversely, the moth brain seems to max out at around 80% accuracy. We don't know if this is due to parameter choice, or if it's an intrinsic limitation of the moth (moths can learn about 8 new odors). For example, honeybee olfactory networks are larger though with the same basic layout, and they demonstrate much richer learning behavior.

#--------------------------------------------------------

To generate insect cyborgs:

	0. The scripts in the folder "insectCyborgScripts" run experiments that use a moth brain's readout neurons as features for use by ML methods. These features substantially improve ML accuracy.

	1. In the folder 'insectCyborgScripts', run 'cyborgExperimentsMainWrapper.m'. Key experiment parameters: numRuns, trPerClassList (in 'runMothLearnerOnReducedMnistForUseInCyborg.m'.
	2. All cyborg-specific scripts and functions are in 'insectCyborgScripts', but the cyborg code calls functions in 'supportFunctions' so this folder must be on path.

#-------------------------------------------------------
 
Many thanks for your interest in these clever moths :) 
We hope you enjoy them! We welcome any questions, comments, (constructive) criticisms, bug reports, improvements, extensions, etc.

Charles Delahunt, delahunt@uw.edu

