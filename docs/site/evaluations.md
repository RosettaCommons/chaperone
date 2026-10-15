# Agent evaluations

The evaluation controller, task prompts, inputs, held-out scorers and reference
answers live in the private
[chaperone-evals repository](https://github.com/Ubiquinone-dot/chaperone-evals).
Access is required. They are not distributed with Chaperone or its public docs.

The private repository's [controller setup guide](https://github.com/Ubiquinone-dot/chaperone-evals/blob/main/controller-project/SETUP.md)
is the source of truth for installation,
site configuration, launching trials and scoring. Install Chaperone from the
revision being evaluated, configure its containers and weights, and then follow
that guide from a separate trusted controller environment. Ordinary Chaperone
workflows do not require Harbor or any evaluation dependencies.

## Isolation

The controller gives the agent only its task prompt, approved inputs and a
filtered snapshot of Chaperone. Worked examples, benchmark solutions, held-out
scorers and the private repository must remain outside the agent workspace and
container mounts. The controller owns scheduler isolation and held-out scoring.
Do not mount the private checkout into an agent container.

## GitHub and Harbor Hub

GitHub stores the maintained source. Harbor Hub dataset releases are separately
uploaded, versioned snapshots; changing GitHub does not update an existing Hub
release. Pin the Chaperone revision, controller revision and dataset version
for a reproducible comparison. The private repository documents how to verify
and restore a dataset snapshot before running a trial.
