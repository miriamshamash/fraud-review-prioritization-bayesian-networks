# Fraud Review Prioritization Using Bayesian Networks

# Overview
This project explores a probabilistic approach to fraud detection that focuses on prioritizing human review rather than making rigid yes-or-no fraud decisions. Using Bayesian Networks, the system explicitly models uncertainty and computes the likelihood of fraud given partial or noisy evidence.

The project builds on a custom Bayesian Network implementation and layers a fraud review workflow on top of it. Instead of collapsing outcomes into a single label, transactions are ranked by estimated fraud risk, allowing reviewers to focus attention where it is most valuable.

# Approach
The core of the system is a Bayesian Network that represents common fraud risk signals, such as unusual transaction amounts, new devices, location mismatches, and prior chargebacks. Given observed evidence for a transaction, the network performs probabilistic inference to compute the posterior probability of fraud.

A separate prioritization layer converts these probabilities into an ordered review queue. This design separates **risk estimation** from **decision-making**, making the system easier to reason about, audit, and adjust as assumptions change.

# Potential Applications
This approach is well suited for financial services, marketplaces, and trust and safety teams where false positives are costly and human judgment remains essential. By ranking cases rather than enforcing hard thresholds, teams can allocate review resources more efficiently while maintaining transparency.

More broadly, the project demonstrates how probabilistic reasoning can support decision workflows in domains where uncertainty is unavoidable and explainability matters.

# Key Concepts
- Bayesian Networks  
- Probabilistic inference under uncertainty  
- Forward enumeration  
- Decision support systems  
- Risk-based prioritization  

# Repository Structure
```text
.
├── BayesNet.py                      # Bayesian Network implementation
├── fraud_review_prioritization.py   # Fraud review prioritization workflow
├── nets/
│   └── fraud_review.json            # Fraud risk Bayesian Network definition
├── README.md

```

# How to Run
``` bash
python fraud_review_prioritization.py
