---
title: "what is constitutional AI?"
date: 2025-05-03
---

[constitutional ai](https://arxiv.org/pdf/2212.08073) trains harmless ai using ai feedback, guided by principles (a 'constitution'), instead of human harm labels.

1.  **supervised phase (initial training):**
    *   generate: get initial responses from a helpful-only model. *purpose: create raw examples.*
    *   critique: ai identifies harm in responses using the constitution. *purpose: find flaws based on principles.*
    *   revise: ai rewrites responses to remove harm, guided by critique. *purpose: create corrected examples.*
    *   finetune: train the model on the ai-revised responses. *purpose: teach the model initial harmlessness.*

2.  **reinforcement learning phase (refinement):**
    *   generate pairs: the supervised model creates pairs of responses to prompts. *purpose: provide choices for comparison.*
    *   ai evaluates: ai chooses the less harmful response based on the constitution. *purpose: create preference data without human labels.*
    *   train preference model (pm): build a model predicting ai preferences. *purpose: capture the constitution as a reward signal.*
    *   reinforce: train the supervised model using the pm's score as a reward. *purpose: optimize the ai for constitutional alignment.*

this process uses ai oversight, defined by a constitution, to scale alignment and reduce reliance on direct human judgment for harmfulness.