# glycosidase design

See the [complete workflow guide](../../docs/site/workflows/enzyme-glycosidase.md) for inputs, chemistry, branching, selection thresholds, runtime prerequisites, and validation limits.

From the repository root:

```bash
python examples/enzyme-glycosidase/pipeline.py --test --rundir=/absolute/path/enzyme-glycosidase-smoke
```

Use `--local --use_local_gpus` inside an allocated GPU session. The bundled recipe requires Foundry, LigandMPNN, and licensed AlphaFold 3 model weights. A one-sample run may reject every design before the later predictors; check per-step artifacts before interpreting it as end-to-end execution.
