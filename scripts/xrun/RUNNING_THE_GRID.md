# Running the grid on CoNLL

5 models x blind arm x 3 replicates = 15 cells of 400 sentences.
Roughly 122 h sequential, ~35 h if you split it across shells.

## Launch

```bash
cd /home/katinska/mobiko_nlp && mkdir -p logs && nohup setsid ./scripts/xrun/run_grid.sh conll blind > logs/grid_conll_driver.log 2>&1 &
```

`nohup setsid` matters: without it a closed terminal SIGHUPs the whole job.

## Resume

**Re-run the identical launch command.** That is the whole procedure — nothing
to pass, nothing to clean up.

- Finished cells (400 records) are skipped.
- A cell killed at sentence 380 continues from 380.
- A sentence that failed is not written, so it is retried.
- A dropped VPN makes the runner *wait* (every 30s, up to `WAIT_MAX`, default
  1h) rather than fail. Tolerate longer with `WAIT_MAX=7200`.

Never run `demo_ag2.py` by hand without `--resume`: it truncates the output file.

## Watch

```bash
cd /home/katinska/mobiko_nlp && tail -f logs/grid_conll_driver.log
```

```bash
cd /home/katinska/mobiko_nlp && for f in data/auto_annotated/grid_conll_*_blind_r*.jsonl; do printf '%-54s %s/400\n' "$(basename "$f")" "$(grep -c '^{' "$f")"; done
```

Per-cell tracebacks are in `logs/grid_conll/<cell>.log`.

## Stop

```bash
cd /home/katinska/mobiko_nlp && pkill -f run_grid.sh; pkill -f demo_ag2.py
```

Kill both — killing the runner alone leaves the current cell running. Whatever
was written is kept and resumes.

## Run models in parallel

`ONLY` filters cells so shells do not race for the same output file:

```bash
cd /home/katinska/mobiko_nlp && ONLY=kimi ./scripts/xrun/run_grid.sh conll blind
```

Values: `qwen`, `apertus`, `deepseek`, `glm`, `kimi`, or `'deepseek\|glm'`.

## Smoke test first (~20 min)

```bash
cd /home/katinska/mobiko_nlp && SMOKE=5 ./scripts/xrun/run_grid.sh conll blind
```

5 sentences per model, r1 only, into `data/auto_annotated/smoke/` so it cannot
be mistaken for a crashed real cell.

## Score

```bash
cd /home/katinska/mobiko_nlp && PYTHONPATH=.:src/multi_agent_annotation .mobiko-venv/bin/python scripts/xrun/eval_blind_pair_conll.py data/auto_annotated/grid_conll_*_blind_r1.jsonl
```

## Notes

- Needs **both** `RCP_API_KEY` and `SWISSAI_API_KEY` in `.env` — Kimi is on a
  different endpoint. Preflight fails loudly if either is missing.
- Commit before launching, or every record is stamped `<sha>-dirty`.
- `DRY_RUN=1` prints the cell list without running anything.
