# Slope RL

This is a clean-slate reinforcement-learning agent for Y8 Slope. It uses
colour-derived road, obstacle, motion, and game-over features from the real
browser canvas, keeps four byte-packed frames of history, and learns the three
actions left/neutral/right with QR-DQN. A model-side extractor converts the
bytes back to normalized values only for sampled training batches.

Slope is endless, so there is no final level to beat. The default success goal
is surviving for 180 seconds; the best checkpoint is selected by deterministic
median survival, with 25th-percentile survival as the tie-breaker.

## Install

Python 3.11 or newer is required. In PowerShell, from this directory:

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Verify the code and installed dependencies without opening the game:

```powershell
python -m unittest discover -s tests -v
```

Check PyTorch and CUDA:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

If an NVIDIA GPU is present but CUDA is false, install the CUDA-enabled
PyTorch build recommended by the [official PyTorch installer](https://pytorch.org/get-started/locally/),
then run the check again.

## Validate the real game first

Run the doctor before committing to a long training job:

```powershell
python slope.py doctor --browser-channel bundled
```

It drives three forced trajectories, checks the observation contract and death
detector, and writes diagnostic frames to `artifacts/doctor`. It must report
`Environment validation passed`. Add `--headed` if you want to watch it:

```powershell
python slope.py doctor --browser-channel bundled --headed
```

Do not train if the doctor reports fewer than two detected deaths. A faulty
terminal signal teaches the model the wrong task.

## Fresh training

Start a new 500,000-transition run on CUDA:

```powershell
python slope.py train --model runs/slope_qrdqn --steps 500000 --device cuda
```

Use `--device auto` to select CUDA when available and otherwise use the CPU.
Training is headless by default; `--headless` may also be supplied explicitly.
Use `--headed` to show the controlled browser while training:

```powershell
# Explicitly headless
python slope.py train --model runs/slope_qrdqn --steps 500000 --device auto --headless

# Visible browser (use a different model path for a separate experiment)
python slope.py train --model runs/slope_qrdqn_visible --steps 500000 --device auto --headed
```

Advertising tabs, pop-under windows, and JavaScript dialogs are blocked or
dismissed in both modes. In-canvas interstitial ads are dismissed as soon as Y8
makes their Close control available. Press `Ctrl+C` once to stop safely; the
latest model and replay buffer are saved before exit.

Use a new `--model` name for a separate experiment. Fresh training refuses to
overwrite an existing latest model or replay unless `--overwrite` is supplied.

## Resume training

A true QR-DQN resume needs both files:

| File | Purpose |
| --- | --- |
| `runs/slope_qrdqn.zip` | Network, optimizer, counters, and exploration state |
| `runs/slope_qrdqn.replay.pkl` | All replayed transitions needed for continued learning |
| `runs/slope_qrdqn_best.zip` | Best deterministic policy for evaluation/play only |
| `runs/slope_qrdqn_best.json` | Best-policy evaluation history |

Resume the latest training state with:

```powershell
python slope.py train --model runs/slope_qrdqn --steps 500000 --resume --device cuda
```

On resume, `--steps 500000` means 500,000 additional transitions. The command
fails deliberately if either the latest `.zip` or `.replay.pkl` is missing.
The `_best.zip` model has no replay buffer and cannot be used with `--resume`.
Copy or back up each latest model/replay pair together.

Keep `--history`, `--fps`, URL, key layout, capture settings, and episode length
the same across training, resume, evaluation, and play. In particular, changing
history changes the model's input shape.

## Evaluate

After the first scheduled evaluation (50,000 transitions by default, using 10
episodes), evaluate the promoted best model over 20 deterministic episodes:

```powershell
python slope.py evaluate --model runs/slope_qrdqn_best --episodes 20 --device cuda
```

To measure the latest training checkpoint instead, use
`--model runs/slope_qrdqn`. Evaluation reports median, 25th percentile, mean,
maximum, and every episode's survival time.

## Play

Watch the best deterministic policy in a visible browser:

```powershell
python slope.py play --model runs/slope_qrdqn_best --device cuda
```

Press `Ctrl+C` to stop. Add `--headless` only when visual playback is not
needed. Before a best checkpoint exists, play `runs/slope_qrdqn` instead.

## Reading training output

The most useful values are:

- `median_survival`: median survival over recent completed exploratory runs.
- `p25_survival`: lower-quartile survival; it exposes brittle policies that
  occasionally run far but usually fail early.
- `Evaluation: median_survival ... p25_survival ...`: deterministic results
  used to promote `_best.zip`; these matter more than exploratory rollout data.
- `ep_len_mean`: mean episode steps. Divide by the control rate (20 by default)
  to estimate seconds.
- `ep_rew_mean`: exploratory reward including the death and action-change
  penalties. It should rise with survival but is not the selection metric.
- `exploration_rate`: random-action probability. It anneals from 1.0 to 0.03
  over the first 30% of the planned run.
- `loss`: the QR-DQN optimization loss. It should remain finite, but lower is
  not automatically better gameplay.
- `throughput` or `fps`: collected transitions per wall-clock second, not the
  game's display refresh rate.

Learning updates begin after 10,000 transitions, so training-loss fields are
not expected before then. Judge progress across several deterministic
evaluations rather than one unusually long run.

## CPU, CUDA, RAM, and disk

The default observation contains 2,608 `uint8` values: 649 colour/geometry
features plus a three-value one-hot action, across four frames. The 150,000-entry
five-step replay stores current and next observations and eventually occupies
about 0.78 GB (0.73 GiB) of host RAM. Its `.replay.pkl` checkpoint is
approximately the same size. While replacing an existing replay checkpoint,
the old and temporary new files can briefly require about 1.6 GB of disk space;
keep at least 3-4 GB free.

Twelve GB of system RAM is a practical minimum and 16-24 GB is preferable when
Chromium is running. The replay lives in system RAM, not GPU memory.

CUDA accelerates the batched QR-DQN network updates, but Chromium rendering,
screenshots, and OpenCV feature extraction remain CPU-bound. GPU use therefore
does not make browser collection run at 60 transitions/s. The default
20-decision/s simulated control rate is intentional. Keep
`--torch-threads 2` initially so Chromium retains CPU time; benchmark 2 versus
4 threads if the machine has spare cores. CPU training remains valid with
`--device cpu`, but updates may take longer.

If Playwright reports that its browser executable is missing, rerun:

```powershell
python -m playwright install chromium
```

## Docker

The Docker image is intended for a VPS without a GPU: it runs Chromium
headlessly and installs CPU-only PyTorch. Its defaults use a 480-pixel capture,
12 control steps per second, and one PyTorch thread so a small CPU does not
starve Chromium. Checkpoints and doctor captures are mounted into the host's
`runs` and `artifacts` directories.

Create the local configuration and build the image:

```bash
cp .env.example .env
docker compose build
```

Validate the live game from inside the container before training:

```bash
docker compose run --rm trainer doctor --browser-channel bundled --headless
```

Start training in the background and follow its console progress:

```bash
docker compose up -d trainer
docker compose logs -f trainer
```

Stop gracefully with enough time to save the model and replay buffer:

```bash
docker compose stop -t 180 trainer
```

Starting the service again automatically resumes when both matching checkpoint
files exist. If only one half exists, startup fails rather than silently
starting an invalid run. Change settings such as `SLOPE_STEPS`,
`SLOPE_MODEL_NAME`, or `SLOPE_REPORT_EVERY` in `.env`, then recreate the
container with `docker compose up -d --force-recreate trainer`.

This default image is CPU-only. A CUDA container additionally requires the
NVIDIA Container Toolkit and a CUDA-enabled PyTorch image/build; setting
`SLOPE_DEVICE=cuda` alone is not sufficient.
