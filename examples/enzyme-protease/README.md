# protease design

See the [complete workflow guide](../../docs/site/workflows/enzyme-protease.md) for inputs, chemistry, branching, selection thresholds, runtime prerequisites, and validation limits.

From the repository root:

```bash
python examples/enzyme-protease/pipeline.py --input_dir=/path/to/authorized/inputs --test --rundir=/absolute/path/enzyme-protease-smoke
```

Use `--local --use_local_gpus` inside an allocated GPU session. The bundled recipe requires Foundry, LigandMPNN, and licensed AlphaFold 3 model weights. A one-sample run may reject every design before the later predictors; check per-step artifacts before interpreting it as end-to-end execution.

Custom structures and chemistry templates are not bundled pending permission.
Supply the authorized files listed in the workflow guide; no jobs are submitted
if `--input_dir` or a required input is missing.
