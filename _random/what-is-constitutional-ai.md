---
title: "What is constitutional AI?"
date: 2025-05-03
description: "Anthropic's way to do RLHF without the 'H'"
---

[Constitutional ai](https://arxiv.org/pdf/2212.08073) trains harmless ai using ai feedback, guided by principles (a 'constitution'), instead of human harm labels.

1.  **Supervised phase (initial training):**
    *   Generate: get initial responses from a helpful-only model. *purpose: create raw examples.*
    *   Critique: ai identifies harm in responses using the constitution. *purpose: find flaws based on principles.*
    *   Revise: ai rewrites responses to remove harm, guided by critique. *purpose: create corrected examples.*
    *   Finetune: train the model on the ai-revised responses. *purpose: teach the model initial harmlessness.*

2.  **Reinforcement learning phase (refinement):**
    *   Generate pairs: the supervised model creates pairs of responses to prompts. *purpose: provide choices for comparison.*
    *   Evaluate: ai chooses the less harmful response based on the constitution. *purpose: create preference data without human labels.*
    *   Train preference model (pm): build a model predicting ai preferences. *purpose: capture the constitution as a reward signal.*
    *   Reinforce: train the supervised model using the pm's score as a reward. *purpose: optimize the ai for constitutional alignment.*

this process uses ai oversight, defined by a constitution, to scale alignment and reduce reliance on direct human judgment for harmfulness.