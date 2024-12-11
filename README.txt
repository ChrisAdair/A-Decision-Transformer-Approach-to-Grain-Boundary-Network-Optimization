This code is for the training and evaluation of the model in "A Decision Transformer Approach to Grain Boundary Network Optimization".
Model was run on Linux systems using PyTorch backend.
---------------------------------
Dependencies:

environment.yml 	- contains the conda environment used for running the training and evaluation
GB5DOF.m		- contains the BRK 5 DoF energy function from "Grain boundary energy function for fcc metals", found in supplemental data 		  			  http://dx.doi.org/10.1016/j.actamat.2013.10.057
		  	(needed only for Evaluation)
MATLAB and MATLAB Compiler - required for running GB5DOF (needed only for Evaluation)
multigame_dt_utils.py 	- contains helper functions from "Multi-Game Decision Transformers" https://github.com/etaoxing/multigame-dt
---------------------------------
Setup:

labGB2BRKEnergy		- requires insertion of GB5DOF.m into this file for evaluating BRK models
environment.yml		- should contain all required python dependencies
MATLAB and MATLAB Compiler - setup as directed by MATLAB

---------------------------------
Training is done through the file experiment.py.

Evaluation is done through the file evaluation.py. Trained models used in the paper are provided in "./Trained Models"
- Options for Evaluation are found in both evaluation.py and GameSimulation.py
- Ensure that evaluation file targets match in both files to either BRK or Linear Evaluation Data

submit_job.sh and submit_job_eval.sh give execution instructions and given resources for each operation.
---------------------------------
Data File Description:

Example: n4-id1Encrypt 1 1-10-2022-04-10.mat

n(Number of Grains)-id(unique mesh/geometry identifier)Encrypt (Level in Game) (Date-Time of Completion).mat

maxScore - Hypothetical maximum property of the given microstructure
structureUsed - Mesh geometries, grain numbers, and required metadata for calculation
tHist - Time history of player decisions: 5 Columns
(Operation) | (Quaternions of Each Grain) | (Raw Material Property) | (Grain ID Changed) | (Position Encoding Eigenvectors)