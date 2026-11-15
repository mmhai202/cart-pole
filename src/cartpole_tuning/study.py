"""Sequential TPE search with deterministic proposals across resumed runs."""

from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic

from .evaluation import Evaluator, PARAMETERS, Q_PARAMETERS
from .persistence import CheckpointWriter


@dataclass
class Search:
    scenarios: list
    validation_scenarios: list
    bounds: list
    output: Path

    @property
    def initial_q(self):
        return self.scenarios[0].q

    @property
    def initial_r(self):
        return self.scenarios[0].r


def run_study(optuna, storage, args, search):
    scenarios, validation_scenarios = search.scenarios, search.validation_scenarios
    initial_q, initial_r = search.initial_q, search.initial_r
    bounds, output = search.bounds, search.output
    training_evaluator = Evaluator(scenarios, args.seeds)
    validation_evaluator = Evaluator(validation_scenarios, args.validation_seeds)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="lqr-tpe", direction="minimize",
                                storage=storage,
                                load_if_exists=args.resume,
                                pruner=optuna.pruners.NopPruner())
    if args.resume:
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    print(f"TPE LQR SEARCH | {len(scenarios)} scenarios × {len(args.seeds)} seeds per trial", flush=True)
    print(f"Searching 4 Q weights + R | Output: {output.resolve()}", flush=True)
    previous = output / "report.json"
    if args.resume and previous.is_file():
        baseline = json.loads(previous.read_text())["baseline_training"]
    else:
        baseline = training_evaluator(initial_q, initial_r)
    print(f"Baseline: unsafe={baseline['unsafe']}/{baseline['runs']}, "
          f"recovered={baseline['runs'] - baseline['unrecovered']}/{baseline['runs']}", flush=True)
    if not args.resume:
        study.enqueue_trial(dict(zip(PARAMETERS, [*initial_q, initial_r])))

    writer = CheckpointWriter(output, scenarios, baseline)

    def objective(trial):
        values = {name: trial.suggest_float(name, low, high, log=True)
                  for name, (low, high) in zip(PARAMETERS, bounds)}
        q, r = [values[name] for name in Q_PARAMETERS], values["r"]
        evaluation = (baseline if q == list(initial_q) and r == initial_r
                      else training_evaluator(q, r))
        trial.set_user_attr("evaluation", evaluation)
        return evaluation["score"]

    def on_trial(study, trial):
        writer.save(study)
        if trial.state.name != "COMPLETE":
            print(f"Trial {trial.number:04d} | {trial.state.name}", flush=True)
            return
        result = trial.user_attrs.get("evaluation", {})
        print(f"Trial {trial.number:04d} | score={trial.value:.5f} | "
              f"unsafe={result.get('unsafe')} | unrecovered={result.get('unrecovered')} | "
              f"errors={result.get('numerical_failures')} | best={study.best_value:.5f}", flush=True)

    started = monotonic()
    try:
        for _ in range(args.trials):
            # Seed by absolute trial number, so resuming follows the same
            # proposal sequence as an uninterrupted sequential run.
            number = len(study.get_trials(deepcopy=False))
            study.sampler = optuna.samplers.TPESampler(
                seed=(args.sampler_seed + number) % 2**32, n_startup_trials=args.startup_trials)
            study.optimize(objective, n_trials=1, callbacks=[on_trial])
        best_q = [study.best_params[name] for name in Q_PARAMETERS]
        best_r = study.best_params["r"]
        print("Checking best weights on held-out seeds...", flush=True)
        validation = {"baseline": validation_evaluator(initial_q, initial_r),
                      "best": validation_evaluator(best_q, best_r)}
        report = writer.save(study, validation, status="complete")
    except KeyboardInterrupt:
        writer.save(study, status="interrupted")
        print(f"\nStopped; checkpoint saved in {output.resolve()}", flush=True)
        return 130
    print(f"Best Q={report['best']['q']} | R={report['best']['r']:g}", flush=True)
    v = validation["best"]
    print(f"Validation: unsafe={v['unsafe']}/{v['runs']}, recovered={v['runs'] - v['unrecovered']}/{v['runs']}", flush=True)
    print(f"Finished in {monotonic() - started:.1f}s; review {output / 'report.json'}", flush=True)
    return 0

