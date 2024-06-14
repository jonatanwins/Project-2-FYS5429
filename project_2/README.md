# Project 2 - Sparse identification of nonlinear dynamics (SINDy) <!-- omit in toc -->

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  
## Introduction

Our Sparse identification of nonlineat dynamics (SINDy) autoencoder project, aims to implement SINDy combined with autoencoders in order to effectively discover sparse representations of dynamical systems. The goal is to provide a framework with focus on modularity, in order for it to be applicable for a wide range of problems. We hope to provide viable results for dynamical systems governed by polynomial and trigonometric differential equations.

A crucial aspect of discovering governing equations is balancing accuracy of the model with descriptive capabilities. Hence it is essential to extract models with the fewest terms required to describe the dynamics of the system. As the number of terms in the model is sensitive to the coordinate system, we include autoencoders, trained simultaneously as the SINDy model, in order to discover the dynamics in a fitting coordinate system. The measurement data is passed through an encoder to find a latent space representation, in which we apply the SINDy algorithm which performs sparce regression sing a library of candidate terms. Finally, this is passed through the decoder, and evaluated against the data in the resulting coordinate system.

We provide a package to apply this framework, with a SINDy library containing polynomial and sine terms. We also provide Jupyter Notebooks where we apply this framework to the Lorenz system and a non-linear pendulum. We utilize a combination of Python3 and JAX to ensure high computation performance.

## Features

We provide an interface for training the SINDy-autoencoder framework on data from dynamical systems, as well as plotting the resulting Xi matrices (i.e. the coefficents for the terms in the discovered equations).

## Installation

The package is setup through calling
```bash
$ pip3 install .
```
in the project_2 directory.

## Usage

The usage is illustrated in the Jupyter Notebooks found in the `src/lorenz` and `src/pendulum` folders.
