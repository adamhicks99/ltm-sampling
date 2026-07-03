# Experiment archive

Each experiment has a numbered, immutable folder:

```text
experiments/
  NNN-short-name/
    README.md       # question, hypothesis, protocol, and acceptance criteria
    config.toml     # exact executable benchmark configuration
    notes.md        # observations, interpretation, and follow-up decisions
    results/        # raw JSONL and paired CSV output
```

Create a new folder when a protocol, dataset suite, model setting, sampler, or
seed policy changes materially. Do not overwrite a completed experiment's
configuration or results. Corrections should be documented in `notes.md`; a
new run with changed methodology gets the next experiment number.

Raw result files remain tracked because they are small and contain the evidence
behind the notes. Large model checkpoints, datasets, and profiler traces remain
excluded by the repository `.gitignore`.
