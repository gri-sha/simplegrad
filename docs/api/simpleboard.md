# SimpleBoard

SimpleBoard is a web dashboard for exploring experiment runs logged by `Tracker`. It reads the SQLite databases written by the tracking module and provides an interactive UI for comparing runs, inspecting metrics over time, viewing hyperparameters, gradient histograms, and computation graphs.

## Launching

SimpleBoard is installed as a CLI command alongside the simplegrad package. Run it from your project directory:

```bash
simpleboard
```

The server starts on `http://127.0.0.1:8000` and opens your browser automatically. By default it looks for experiment databases in `./experiments/`.

## Options

```bash
simpleboard [--port PORT] [--host HOST] [--all-exp-dir DIR] [--no-browser]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port`, `-p` | `8000` | Port to bind the server to. |
| `--host` | `127.0.0.1` | Host address to bind to. Use `0.0.0.0` to expose on the network. |
| `--all-exp-dir`, `-e` | `./experiments` | Directory containing experiment SQLite databases. |
| `--no-browser` | — | Start the server without opening the browser. |

## Examples

```bash
# default — serves on port 8000, opens browser
simpleboard

# custom port
simpleboard --port 8080

# point to a different experiments directory
simpleboard --all-exp-dir ./runs

# headless server (useful on remote machines)
simpleboard --host 0.0.0.0 --no-browser
```

## Using with Tracker

Point SimpleBoard at the same directory you passed to `Tracker` when logging:

```python
import simplegrad as sg

tracker = sg.Tracker(all_exp_dir="./experiments")
tracker.set_experiment("mnist")
tracker.start_run("baseline", hparams={"lr": 0.01, "epochs": 20})

for epoch in range(20):
    # ... training ...
    tracker.log({"loss": loss_val, "acc": acc_val}, step=epoch)

tracker.end_run()
```

Then launch the dashboard:

```bash
simpleboard --all-exp-dir ./experiments
```

SimpleBoard will list all experiments and runs. Select a run from the sidebar to plot its metrics, compare against other runs, inspect hyperparameters, and browse any logged computation graphs or histograms.

## Frontend source

The dashboard frontend lives in `simplegrad/simpleboard/app/`. After making changes to the frontend source, rebuild the compiled assets before committing:

```bash
python scripts/build_web.py
```
