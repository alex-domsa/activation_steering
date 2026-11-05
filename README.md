# Activation Steering and Analogous Pairs

Final-year project, BSc Computer Science and Software Engineering, Maynooth University.
Supervisor: Dr. Diarmuid O'Donoghue.

**Question.** Can contrastive activation methods (CAA, Rimsky et al. 2024) find a direction in a language model's
activations that distinguishes analogous pairs, and a steering vector with evidence that it does what we claim?

## Approach
1. **Data.** StoryAnalogy story pairs, grouped by their human ratings:
   *analogy* (relation >= 2, entity <= 1) vs *unrelated* (relation <= 1, entity <= 1). Entity similarity is matched, so the
   groups differ only in whether they share a relation. Balanced by domain and sentence length; train / validation / test
   split with no shared sentences. The test split is used once, at the end.
2. **Directions.** Cache the hidden state of the final token at every layer; a direction is the mean activation of one group
   minus the mean of the other (computed on the training split).
3. **Validation (five tests).** A direction must
   1. separate the groups on held-out data, better than random directions (AUC);
   2. not be explained by surface features (sentence length, word overlap, sentence structure, word-level classifiers);
   3. not be the model's "yes"/answer direction;
   4. appear across prompt formats;
   5. be causal and specific: adding it changes the intended behaviour, more than random vectors, and only that behaviour.
4. **Steering.** Add the vector to the residual stream at one layer with forward hooks (PyTorch + Hugging Face), measure the
   change in the model's behaviour against random vectors of the same length.

## Repository layout
```
src/analogy_steering/   shared code: config, data loading, prompts, model hooks, metrics, directions
experiments/            numbered scripts, run in order (01 ... 09)
data/                   datasets (not committed, see data/README.md)
results/summary/        final result tables and plots
docs/                   references and the experiment log
```

## Setup
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes src/analogy_steering importable
```
Model and paths are set in `src/analogy_steering/config.py` (default: `Qwen/Qwen2.5-1.5B-Instruct`, runs on an M2 MacBook via MPS).

## Status
Work in progress. Results so far (quick runs, Qwen2.5-1.5B-Instruct) are recorded in `docs/experiment_log.md`.
