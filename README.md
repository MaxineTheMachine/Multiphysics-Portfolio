# Automated Multiphysics Simulation of MEMS Cantilever and Integrated Optical Waveguides
Sections
Problem overview


# We model:

A silicon waveguide fabricated on top of a MEMS cantilever

MEMS deflection causes:

Rigid translation of the waveguide center

(Optional, later) strain-induced index perturbation

Optical propagation is recomputed on the deformed geometry

This exact abstraction is used in:

Tunable photonic MEMS

Phase shifters

Optomechanical sensors



# This single demo shows:

Multiphysics coupling

PDE solvers

Geometry deformation

Optical mode sensitivity

Software architecture

digital twin workflow:

Layer	Purpose
Simulation	    |       Ground truth
Surrogate	|       Fast approximation
Active learning	|       Smart sampling
Bayesian optimization   |	Design targeting
Physics-informed features	|   Structure & generalization


Physics intuition






# Physics models


# Software architecture


# Example results


# How to run


# Extensions (ML surrogates, optimization)


A Gaussian process surrogate model was trained to replace repeated multiphysics MEMS–photonics simulations, achieving sub-percent prediction error while reducing evaluation time by orders of magnitude.







