# Data-for-social-reinforcement-learning
Data for the paper 'How social reinforcement learning can lead to metastable polarization and the voter model.' authored by Benedikt V Meylahn and Janusz M Meylahn available at https://arxiv.org/abs/2406.07993 .

The paper investigates the use of Q-learning as a basis for opinion dynamics. 

Abstract:
Previous explanations for the persistence of polarization of opinions have typically included modelling assumptions that predispose the possibility of polarization (i.e., assumptions allowing a pair of agents to drift apart in their opinion such as repulsive interactions or bounded confidence). An exception is a recent simulation study showing that polarization is persistent when agents form their opinions using social reinforcement learning.
Our goal is to highlight the usefulness of reinforcement learning in the context of modeling opinion dynamics, but that caution is required when selecting the tools used to study such a model. We show that the polarization observed in the model of the simulation study cannot persist indefinitely, and exhibits consensus asymptotically with probability one. By constructing a link between the reinforcement learning model and the voter model, we argue that the observed polarization is metastable. Finally, we show that a slight modification in the learning process of the agents changes the model from being non-ergodic to being ergodic.
Our results show that reinforcement learning may be a powerful method for modelling polarization in opinion dynamics, but that the tools (objects to study such as the stationary distribution, or time to absorption for example) appropriate for analysing such models crucially depend on their properties (such as ergodicity, or transience). These properties are determined by the details of the learning process and may be difficult to identify based solely on simulations.

The data in this repository is underlying Figures 2 and 3. Furthermore we provide the base code for the simulation (with and without plotting).
