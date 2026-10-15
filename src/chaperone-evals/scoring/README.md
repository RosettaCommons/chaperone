# scoring

Shared machinery for the held-out scorers: `SequenceSubmissionSpec` (the
submission contract), `run_sequence_scorer` (fold, measure, gate, report), and
`define_score_flags`.

Nothing an agent runs imports this package, and the harness bundle never includes
it. It is generic: the scorers that apply a task's gates, the reference submissions and
the calibration data are held out in the private tree.

In Harbor runs the scorer receives `trial/controller/run_meta.json`, generated
after the agent-side guarded Slurm registry closes. It contains actual
allocation time reconciled from `sacct` (with Chaperone `.execution` records as
a fallback), so held-out scoring jobs are excluded. Submission-provided cost,
lineage, terminal-report, and public-score files are ignored. Missing transcript
or incomplete accounting is recorded as an audit warning and never suppresses
scoring of a readable sequence table. Direct standalone scorer use may still
provide an external `--run_meta` file.
