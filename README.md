
23 April 2018

This repository contains the code used in the paper "Putting a bug in ML: The moth olfactory network learns to read MNIST" (CB Delahunt and JN Kutz, January 2018).

This codebase simulates a moth olfactory network as it learns to read MNIST digits. It generates a model of the Manduca sexta moth olfactory network, then runs a time-stepped evolution of SDEs to train the network to read downsampled MNIST digits. The moth's architecture is able to learn very rapidly (1 to 10 digits per class).
Notes:

1. 'runMothLearnerOnReducedMnist.m' runs the simulation. Key experiment parameters: goal, trPerClass, numSniffs.

2. Run time for a simulation is 30 - 120 seconds. It is a time-stepped simulation of a fast learning system (the moth brain), not a fast system in itself.

3. The learning task is not MNIST per-se. Rather, it is a simpler, cruder version of MNIST, created by downsampling then selecting a subset of pixels. The features are pixels (no spatial relationships). This is done to accommodate the size of the natural moth brain architecture (~ 60 features).

4. 'runOtherAlgorithms' contains scripts to run Nearest-neighbor, SVM, and neural net on the downsampled MNIST. Key experiment parameters are: trPerClass, numNeighbors, boxConstraint (matlab script); trainPerClass (python script).

5. We find that the moth brain out-performs the ML methods given in the few-samples regime (less than 20 training samples per class). Conversely, the moth brain seems to max out at around 85% accuracy. We don't know if this is due to the learning rates we chose, or if it's an intrinsic limitation of the moth architecture at moth size (moths can learn about 8 new odors). For example, honeybee olfactory networks are larger, though with the same basic layout. They demonstrate much richer learning behavior.
 
6. We had earlier found that the moth brain substantially out-performed ML methods. However, this was due to a bug (...) in the pre-processing code that unfairly penalized ML methods. So the ML methods come closer to the moth performance than we previously thought, and match moth performance at 20 training samples per class. The ML methods given here can likely be optimized.

7. The code here is a variant, for MNIST experiments, of code at github/charlesDelahunt/smartAsABug, which supports the paper "Biological Mechanisms for Learning: A Computational Model of Olfactory Learning in the Manduca sexta Moth, with Applications to Neural Nets" (CB Delahunt, JA Riffell, JN Kutz, January 2018). The mechanics of the moth architecture and the SDE evolution are largely the same.

Many thanks for your interest in these clever moths :) 
We hope you enjoy them, and we welcome any questions, comments, (constructive) criticisms, bug reports, improvements, extensions, etc.

Charles Delahunt, delahunt@uw.edu

