# FEWTrax
The purpose of this package is to generate fast EMRI tracks corresponding to a single harmonic mode for EMRIs. 

## Context
When searching for EMRI signals in LISA data, a common approach is a single mode search. Here a semi-coherent search statistic using STFTs is used to identify chirping tracks in the time-frequency plane. This single track only tells us about the frequency evolution of this trackm which is related to the evolution of the orbital parameters. Since only one track does not suffice to pinpoint the intrinsic parameters well, we aim to search for multiple tracks given that we have a goo proxy to the found one. By exploiting the fact that the frequency evolution of all tracks is governed by integer combinations of the three fundamental frequency evolutions, we hope to find more tracks and identify the mode numbers of the found tracks. If we have this, we can narrow down the expected amplitudes for each of the tracks and use this for a more sensitive identification towards the true parameters of the syste. Context can be found in the papers here [](https://arxiv.org/abs/2510.20891), [](https://arxiv.org/abs/2506.09470), [](https://arxiv.org/abs/2510.19047).

## Task 1
Identify the suitability of the current fewtrax implementation for the intended purpose. Pay particular attention to the accuracy of generated tracks and to possibility to generate tracks for single modes. 

## task 2 
Identify possible bottlenecks in a AJX-implementation of this idea. Research gradient-based approaches for tackling this multi-mode search problem. 

## Task 3
Implement integration backwards from the found track. For plunging systems, the advantage is that we determine the time of plunge and can integrate backwards from there. This is a more stable approach than integrating forward from the initial conditions, which are more uncertain.

## Task 4
Research and outline a gradient-based approach to identify the parameters of the system given the found tracks. This will likely involve defining a loss function based on the difference between the generated tracks and the found tracks, and then using an optimization algorithm to minimize this loss function with respect to the system parameters. Write this down in a markdown file, with links to relevant repositories and algorithms. 

Before implementing the tasks above, outline the steps you would take to complete each task. Mention advantages and disadvantages of different approaches you might take for each task. Assess the overall feasability of the project and identify any potential challenges or limitations that may arise during the implementation. Pay atention to memeory concerns: vmap and parallel sampling may be memory-prohibitive. Consider returning tracks as sparse arrays, as they evolve smoothly in time-frequency and cubic splines likely suffice to capture thr evolution. 



