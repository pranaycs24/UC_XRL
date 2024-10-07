Unit Commitment Problem using XRL (Explainable Reinforcement Learning)
Project Overview
This project implements a solution to the Unit Commitment Problem (UCP) using Explainable Reinforcement Learning (XRL). The Unit Commitment Problem is a fundamental optimization challenge in power systems, which involves determining the optimal schedule for turning power generation units on and off to meet demand while minimizing operational costs.

In this approach, we apply Reinforcement Learning (RL) to optimize the scheduling, while explainability techniques provide insights into the decision-making process, making the RL model interpretable and ensuring the results are trustworthy for operators and regulators.

Features
Unit Commitment Optimization: Solves the classic Unit Commitment Problem using RL.
Explainable Reinforcement Learning (XRL): Provides transparency into how decisions are made by the RL agent.
Flexible Cost Function: Supports a customizable cost function, including fuel costs, start-up/shutdown costs, and emissions.
Scalable Framework: Can be adapted to small- or large-scale power systems with varying numbers of generation units.
Installation
Requirements
Python 3.x
Key Python libraries:
pytorch or tensorflow (depending on the deep RL framework)
gym
matplotlib
numpy
pandas
scikit-learn
lime or shap (for explainability)
