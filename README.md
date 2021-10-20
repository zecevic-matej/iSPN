### Interventional Sum-Product Networks (iSPN)

Code repository for the corresponding paper in the *35th Conference on Neural Information Processing Systems (NeurIPS 2021)*.

> **Abstract**
> While probabilistic models are an important tool for studying causality, doing so suffers from the intractability of inference. As a step towards tractable causal models, we consider the problem of learning interventional distributions using sum-product networks (SPNs) that are over-parameterized by gate functions, e.g., neural networks. Providing an arbitrarily intervened causal graph as input, effectively subsuming Pearl’s do-operator, the gate function predicts the parameters of the SPN. The resulting interventional SPNs are motivated and illustrated by a structural causal model themed around personal health. Our empirical evaluation on three benchmark data sets as well as a synthetic health data set clearly demonstrates that interventional SPNs indeed are both expressive in modelling and flexible in adapting to the interventions.

![Showing iSPN and how it adequately captures Causal change.](media/Figure_Motivation.png)

#### Instructions

Make sure you have installed all necessary dependencies.

The code relies on TensorFlow. While some of the Baselines are using the PyTorch backend.

To reproduce the density estimation results, run `.iSPN/iSPN_for_causal_toy_dataset_interventions_joint_dist_cont.py`.

To reproduce the ATE estimation results, run `.iSPN/iSPN_for_causal_toy_dataset_interventions_joint_dist_cont_ate_benchmark.py`.

