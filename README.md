# Slope AI (no training required)

This is a Python vision controller for browser-based versions of **Slope**. It
looks at the game canvas on your screen, estimates the green track and red
obstacles, then presses the left/right keys. It does not need a dataset, a saved
model, access to the game's code, or a training run.

The project now has two modes:

- `slope_ai.py` is the original controller and plays immediately without training.
- `slope_rl.py` is a PPO reinforcement-learning agent that learns by playing.

## Install

Python 3.10 or newer is recommended. In PowerShell, from this folder:

```powershell
python -m pip install -r requirements.txt
```

## One-time setup (desktop capture only)

1. Open Slope in your browser and make the game fully visible. A large window
   with browser zoom at 100% works best.
2. Run:

   ```powershell
   python slope_ai.py --setup
   ```

3. Draw a tight rectangle around the game canvas—not the browser tabs or address
   bar—and press Enter. The coordinates are saved in `slope_ai_config.json`.

Run setup again after moving or resizing the browser window.

This setup is used by `slope_ai.py` and by the legacy desktop-capture mode of
`slope_rl.py`. URL mode launches and captures its own browser instances, so it
does not use `slope_ai_config.json`.

## Play

```powershell
python slope_ai.py
```

Focus the browser and start/restart the game yourself. The controls are global,
so the terminal does not need focus:

- **F8** — start or pause the AI
- **F9** — quit immediately and release the steering keys

Most versions use arrow keys. For a version that uses A/D, run:

```powershell
python slope_ai.py --keys ad
```

To inspect what it detects, add `--preview`. The white circle is the detected
ball, the yellow line is its target, and a magenta box is the obstacle it is
avoiding:

```powershell
python slope_ai.py --preview
```

Do not click the preview while playing—the game tab must keep keyboard focus.

## Getting better results

- Use a clean, unobstructed game canvas and hide video overlays or ads.
- Keep a steady window size and rerun `--setup` whenever it changes.
- A 16:9 canvas of at least 800×450 works best.
- If capture is slow, try `python slope_ai.py --fps 20`.
- Different Slope clones use different colors and physics. The included detector
  accepts a broad neon-green range, but the steering constants may need tuning
  for a substantially different clone.

This is a general visual bot rather than a guaranteed perfect solver. It will
play immediately without training, but random/generated tracks can still beat a
heuristic controller.

## Reinforcement-learning mode

Install the additional packages:

```powershell
python -m pip install -r requirements-rl.txt
```

### Parallel URL mode (recommended)

URL mode launches independent Playwright browser processes. Each worker owns its
page, canvas capture, keyboard input, and game clock, so the games cannot steal
focus or send controls to one another. No region setup, five-second focus
countdown, or manually focused browser window is needed.

For the Danish Y8 Slope page, train four games at once with:

```powershell
python slope_rl.py train --url "https://da.y8.com/games/slope" --instances 4 --steps 100000
```

Y8 game-page URLs such as this one are automatically changed to Y8's lightweight
embed URL. The automated session starts and restarts the game itself. It also
rejects Y8's optional privacy prompt and, when a pre-roll appears, waits for and
uses the ad provider's explicit close control. An ad can therefore delay a
worker's startup. Transient ad-control failures trigger a bounded browser
relaunch inside that worker instead of immediately stopping the full run.

Google Chrome is the default browser channel. If Chrome is not installed, use
Microsoft Edge:

```powershell
python slope_rl.py train --url "https://da.y8.com/games/slope" --instances 4 --steps 100000 --browser-channel msedge
```

Alternatively, install Playwright's bundled Chromium once and select it:

```powershell
python -m playwright install chromium
python slope_rl.py train --url "https://da.y8.com/games/slope" --instances 4 --steps 100000 --browser-channel bundled
```

Automated browsers are headless by default. To watch just the first game while
the other three continue headlessly, use:

```powershell
python slope_rl.py train --url "https://da.y8.com/games/slope" --instances 4 --visible-instances 1 --steps 100000
```

Use `--headed` instead to show every browser window. Visible windows still
receive isolated input and do not need desktop focus. Their virtual game clock
is paced in display-sized slices so movement remains watchable; because vector
workers synchronize each step, a visible instance limits collection to roughly
real-time speed. Leave all instances headless for maximum throughput. Unexpected
advertising popup pages are closed automatically.

`--steps` counts aggregate environment transitions, not transitions per game.
For example, 100,000 steps with four instances is roughly 25,000 transitions
from each worker. PPO rollouts are sized to stay near 512 aggregate samples.
Likewise, `--checkpoint-every 10000` (the default) means approximately every
10,000 aggregate transitions across all instances, and not 10,000 from every
worker. Parallelism reduces collection time; it does not multiply the requested
training sample count.

Each instance adds a browser process, a Python worker, canvas screenshots, and a
WebGL game. Start with two to four instances and watch CPU, RAM, GPU load, and
actual steps per second. More workers can become slower or less stable once the
machine is saturated. `--fps` sets the action cadence for each game; it does not
guarantee that the host can process that many steps per second.

Press **F9** during training to stop safely after the current vector step. The
latest model is then saved to `models/slope_ppo.zip` (or the path passed to
`--model`). Periodic checkpoints are written under `models/` as well.

After at least one episode finishes, PPO's training table includes
`rollout/ep_rew_mean` and `rollout/ep_len_mean`. Rising values generally mean
the policy is earning more reward and surviving for more steps. Compare trends
over several iterations rather than treating one rollout as conclusive.

Resume the same parallel run with:

```powershell
python slope_rl.py train --url "https://da.y8.com/games/slope" --instances 4 --steps 100000 --resume
```

Run the trained policy in one isolated browser without further learning:

```powershell
python slope_rl.py play --url "https://da.y8.com/games/slope"
```

Add `--visible-instances 1` to watch playback. Visible play mode leaves the
game clock running continuously for smooth animation; training remains
clock-controlled and deterministic.

### Legacy desktop-capture mode

Desktop mode still controls one already-open browser using global keyboard and
mouse input. Complete the one-time `--setup`, keep the game at the configured
screen position, and start training with:

```powershell
python slope_rl.py train --steps 100000
```

The script waits five seconds so you can focus the browser. At 12 FPS, 100,000
steps takes at least 2.3 hours plus restarts and PPO updates; `--fps 60` has a
theoretical minimum of about 28 minutes if the machine and game sustain it. The
computer must remain awake, the game must stay focused, and the window must not
move. Desktop mode supports only one instance; `--instances` greater than one
requires `--url`.

By default it gives PPO 15 normalized features already extracted by the colour
detector: ball and target positions, their motion, track confidence/density,
obstacle geometry, and the last steering direction. This trains a small MLP
much faster on a CPU than the old stacked-image CNN. It rewards staying alive
and aligned with the detected track, gives a penalty for crashing, and presses
Space to restart.
Pressing **F9** has the same safe-stop-and-save behavior described above.

If your game restarts with Enter or R, use one of these:

```powershell
python slope_rl.py train --steps 100000 --restart-key enter
python slope_rl.py train --steps 100000 --restart-key r
```

If the game has an on-screen **AGAIN** button that does not reliably respond to
the keyboard, leave the game on that death screen and run:

```powershell
python slope_rl.py train --steps 100000 --restart-click
```

During the countdown, move the mouse pointer over **AGAIN** without clicking it,
then leave the pointer there for the entire run. The script clicks the current
mouse position after every death. Keep the browser window in the same position
while training. This mode also detects the static game-over canvas used by the
Y8 Unity version, even when the green track remains visible behind the overlay.
Static detection compares colour snapshots a quarter-second apart and requires
about one second of confirmed stillness, preventing high-FPS gameplay with small
frame-to-frame changes from being mistaken for a death screen.

Resume learning from the saved model:

```powershell
python slope_rl.py train --steps 100000 --resume
```

Run the trained policy without further learning:

```powershell
python slope_rl.py play
```

The agent chooses left, straight, or right. To train the original image policy,
pass `--observation camera`; it sees four consecutive 84×84 RGB frames. Camera
and feature checkpoints are not interchangeable, so also use a separate model
path, such as `--model models/slope_ppo_camera`. Because Slope websites differ,
use the restart-key option that matches your version and rerun region setup
after moving the browser when using legacy desktop mode.

## Docker Compose on a CPU VPS

The Compose service runs URL mode in its own headless Chromium browser. A VPS
cannot capture or control a Chrome window on your local computer. The VPS only
needs Docker Engine with Compose; no GPU or NVIDIA runtime is required.

Create the local settings file, build the image, and start training:

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f trainer
```

After pulling code changes, always rebuild and replace the existing container so
the Compose arguments and the Python code in the image stay in sync:

```bash
docker compose down
docker compose build --pull
docker compose up -d --force-recreate
docker compose logs -f trainer
```

An `unrecognized arguments` message for options present in `compose.yaml` means
the service is still running an older image. The rebuild above fixes that
version mismatch. The entrypoint automatically chooses a free virtual display,
so stale Xvfb locks left by a failed restart no longer break the next attempt.

The first build downloads bundled Chromium and CPU-only PyTorch. Defaults in
`.env.example` train one game for 100,000 aggregate steps and resume
`models/slope_ppo_features.zip` when it exists. The VPS profile uses the compact
feature policy, a 12 Hz action cadence, a 480-pixel viewport, JPEG capture, and
one PyTorch thread. New feature policies receive a short synthetic warm start
from the proven steering rule before PPO exploration begins. These settings
reduce software WebGL, screenshot, and neural network cost while preserving
enough detail for the colour detector. Change
`SLOPE_INSTANCES`, `SLOPE_STEPS`, `SLOPE_FPS`, `SLOPE_CAPTURE_WIDTH`,
`SLOPE_MODEL`, or `CHECKPOINT_EVERY` in `.env` as needed.

Old `slope_ppo.zip` camera checkpoints cannot be resumed by this feature policy.
Keep them under a different name and use `--observation camera` when playing or
continuing them.

Confirm that the container has the CPU-only PyTorch build while it is running:

```bash
docker compose exec trainer python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Stop it gracefully with:

```bash
docker compose stop
```

The container handles the stop signal after its current vector step and saves
the latest model. The host `./models` directory is mounted at `/app/models`, so
models and checkpoints survive image rebuilds and container replacement. A
normally completed training run remains stopped; run `docker compose up -d`
again to resume for another configured number of steps.

The latest resumable policy is `models/slope_ppo_features.zip`. After ten
episodes, training also maintains `models/slope_ppo_features_best.zip` whenever
the rolling mean episode reward reaches a new high. Use the `_best` file for
playback and keep the regular file for continuing the latest training state.

The default image uses CPU-only PyTorch 2.14. Start with one browser instance on
a two-vCPU host. Increase `SLOPE_INSTANCES` only while there is spare CPU and
RAM; more workers can reduce throughput after the machine is saturated. PPO's
logged `fps` is aggregate environment steps per wall-clock second, not the
configured game cadence. Judge learning from the trend in `ep_len_mean` and
`ep_rew_mean` over several rollouts. A `Performance:` line is printed every
minute with measured steps/second plus browser and vision milliseconds per step;
this shows which part is limiting the VPS.
