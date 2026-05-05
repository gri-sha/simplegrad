# Tracking

The experiment tracking module lets you log metrics, hyperparameters, and computation graphs across training runs, storing everything in a SQLite database. `Tracker` is the high-level API: you set an experiment, start a run, log scalars after each epoch, and end the run. Logged data can then be explored via the SimpleBoard web dashboard.

```python
import simplegrad as sg

tracker = sg.Tracker(all_exp_dir="./experiments")
tracker.set_experiment("mnist_run")
tracker.start_run(run_name="baseline", hparams={"lr": 0.01, "batch_size": 32})

for epoch in range(10):
    loss = ...  # compute loss
    tracker.log({"train_loss": float(loss.values)}, step=epoch)

tracker.end_run()
```

::: simplegrad.track.tracker.Tracker

---

::: simplegrad.track.exp_db_manager.ExperimentDBManager

::: simplegrad.track.exp_db_manager.RunInfo

::: simplegrad.track.exp_db_manager.RecordInfo

---

::: simplegrad.track.comp_graph._build_graph_data
