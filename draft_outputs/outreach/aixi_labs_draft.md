Subject: Finite-state benchmark of AIXI-style planning and simple learners (draft)

Hello,

I built a small local benchmark that compares a tabular Q-learner, an AC-CTW-UCT agent (inspired by MC-AIXI-CTW, not a
faithful reproduction), an exact dynamic-programming oracle, a random policy and a local 0.5B policy-table model on 20
frozen finite-state tasks. Task difficulty is described by a separate proxy vector, not by Kolmogorov complexity.

Suite-level results with intervals are in `README_draft.md` of the project folder. I would value your view on whether
the task families and the proxy vector are a sensible way to organize this kind of comparison.

Best regards,
Ajinkya

Draft only. Recipient and contact details intentionally left unset.
