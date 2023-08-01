# QL_challenge

Vasu Patel has attempted to solve the QL challenge (create an interactive human-in-the-loop training system using a simple machine learning model) for MLE role between Thu, Jul 27, 2023 - Wed, Aug 2, 2023.

## Key functionalities
1. Frontend allows users to interactively add data points to the graph.
2. Backend accept these data points and train the Gaussian Mixture Model (GMM).
3. Bacend return the GMM parameters to the front-end for plotting GMM.

## Steps
1. Create a python venv using `python -m venv ./ql_venv` in the root directory (i.e. `QL_challenge`).
2. Activate venv using `source ./ql_venv/bin/activate`.
3. Install requirements listed in requirements.txt using `pip install -r requirements.txt`.
4. From local machine, run `uvicorn solution_skeleton:app --reload`.
5. In the web application, choose the value of `n_components` between and including [2, 10].
6. Enter data points by clicking in the input window.
7. Press `p` for training model with input data points and extracting model parameters to generate a graph.
8. This repository implements model versioning by saving the trained model(s) in the root directory (i.e. `QL_challenge`).

## Assumptions, Q&A
Assumptions, Q&A related to judgement, and clarifying question can be found in `./Q_and_A.txt`.