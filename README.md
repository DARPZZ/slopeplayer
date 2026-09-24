# Slope RL

This is a clean-slate reinforcement-learning agent for Y8 Slope. Each frame
of the real browser canvas is shrunk to a 64x40 image of its red and green
intensity (Slope draws the road, ball, and buildings in green and deadly walls
in red). Four byte-packed frames of history, with the action behind each,
feed a small CNN that learns the three actions left/neutral/right with
QR-DQN. The bytes are converted to normalized values only for sampled
training batches.

Slope is endless, so there is no final level to beat. The default success goal
is surviving for 180 seconds; the best checkpoint is selected by deterministic
median survival, with 25th-percentile survival as the tie-breaker.

A run ends at the first of two death signals:

- **Frozen screen.** Slope's camera never stops while the ball is alive. About a
  second after a crash the scene freezes, roughly 2.6 s before the death screen
  appears. Ending the episode here puts the penalty close to the mistake that
  caused it.
- **Death screen.** The AGAIN, Menu, and Leaderboard controls are matched
  against templates taken from real captures (`slope_core/assets`), so the
  check works at any capture width and JPEG quality.

Exploration holds each random action for 2–8 steps, rather than picking a new
one every step, so random play actually tries different paths through turns.

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
detector, and writes diagnostic frames to `artifacts/doctor`. Each line names
the signal that ended the run (`frozen_screen` or `retry_screen`). It must report
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

### Parallel browsers

Browser collection is the bottleneck, so `--envs` runs several browsers at
once, each in its own worker process:

```powershell
python slope.py train --model runs/slope_qrdqn --steps 500000 --envs 3 --device cuda
```

`--steps` counts transitions from all browsers together, and the network still
makes one update per transition. Each extra browser needs roughly one CPU core
and 0.5-1 GB of RAM; the replay size does not change. A run must be resumed
with the same `--envs` value it was started with, because the replay keeps a
separate transition sequence per browser.

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

On resume, `--steps 500000` means 500,000 additional transitions. The
exploration schedule is fixed when the run starts and is not affected by
the resume `--steps` value. The command
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
- `exploration_rate`: probability of starting a held random action. It
  anneals from 1.0 to 0.01 over the first 10% of the fresh run's `--steps`.
  That length is stored in the checkpoint, so resuming never raises
  exploration again.
- `loss`: the QR-DQN optimization loss. It should remain finite, but lower is
  not automatically better gameplay.
- `throughput` or `fps`: collected transitions per wall-clock second, not the
  game's display refresh rate.

Learning updates begin after 10,000 transitions, so training-loss fields are
not expected before then. Judge progress across several deterministic
evaluations rather than one unusually long run.

## CPU, CUDA, RAM, and disk

The default observation contains 20,492 `uint8` values: a 2x40x64 red/green
image plus a three-value one-hot action, across four frames. The 100,000-entry
five-step replay stores current and next observations and eventually occupies
about 4.1 GB (3.8 GiB) of host RAM. Its `.replay.pkl` checkpoint is
approximately the same size. While replacing an existing replay checkpoint,
the old and temporary new files can briefly require about 8.2 GB of disk space;
keep at least 12 GB free.

Sixteen GB of system RAM is a practical minimum and 24 GB or more is
preferable when several Chromium browsers are running. The replay lives in system RAM, not GPU memory.

CUDA accelerates the batched QR-DQN network updates, but Chromium rendering,
and screenshots remain CPU-bound. GPU use therefore
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
