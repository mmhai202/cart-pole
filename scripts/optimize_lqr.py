"""Tune diagonal LQR weights with Optuna TPE on the full nonlinear simulator."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy

from configuration import load_config
from cartpole_tuning.evaluation import Scenario, q_parameters
from cartpole_tuning.persistence import write_json
from cartpole_tuning.study import Search, run_study


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("configs", type=Path, nargs="*", default=[Path("config.single.json")],
                        help="Training scenarios; first config supplies starting Q and R")
    parser.add_argument("--trials", type=int, default=100, help="Additional trials to run, including the initial baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Shared training noise seeds")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[1042, 1043, 1044])
    parser.add_argument("--validation-configs", type=Path, nargs="+", help="Unseen scenarios; defaults to training scenarios")
    parser.add_argument("--search-factor", type=float, default=100, help="Search each initial Q entry and R divided/multiplied by this factor")
    parser.add_argument("--startup-trials", type=int, default=10, help="Random startup trials before adaptive TPE")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="New result directory; defaults to results/lqr-tpe-TIMESTAMP")
    parser.add_argument("--resume", action="store_true", help="Continue a matching study in --output")
    return parser


def validate_options(args):
    """Reject invalid search options before creating any output."""
    if args.trials < 1 or args.startup_trials < 0 or not 0 <= args.sampler_seed < 2**32:
        raise ValueError("Trials must be positive; startup trials nonnegative; sampler seed in [0, 2**32).")
    if not np.isfinite(args.search_factor) or args.search_factor <= 1:
        raise ValueError("--search-factor must be finite and greater than 1.")
    for seeds in (args.seeds, args.validation_seeds):
        if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("Noise seeds must be distinct nonnegative integers.")
    if set(args.seeds) & set(args.validation_seeds):
        raise ValueError("Training and validation seeds must not overlap.")
    if args.resume and args.output is None:
        raise ValueError("--resume requires --output.")


def prepare_search(args):
    """Resolve scenarios and a finite logarithmic search space."""
    scenarios = [Scenario(path, load_config(path)) for path in args.configs]
    validation_scenarios = ([Scenario(path, load_config(path)) for path in args.validation_configs]
                            if args.validation_configs else scenarios)
    if len({s.plant.state_size for s in [*scenarios, *validation_scenarios]}) != 1:
        raise ValueError("All training and validation scenarios must use the same number of poles.")
    initial_q, initial_r = scenarios[0].q, scenarios[0].r
    bounds = [(value / args.search_factor, value * args.search_factor) for value in [*initial_q, initial_r]]
    if any(not (0 < low < high < float("inf")) for low, high in bounds):
        raise ValueError("Search bounds must be finite and positive.")
    output = args.output or Path("results") / datetime.now().astimezone().strftime("lqr-tpe-%Y%m%d-%H%M%S-%f")
    return Search(scenarios, validation_scenarios, bounds, output)


def prepare_output(args, search, optuna_version):
    """Create a new manifest or verify provenance before resuming."""
    scenarios, validation_scenarios = search.scenarios, search.validation_scenarios
    initial_q, initial_r = search.initial_q, search.initial_r
    bounds, output = search.bounds, search.output
    sources = sorted({
        "src/simulation.py", "src/configuration.py", "src/experiment.py", "scripts/optimize_lqr.py",
        *(str(path.relative_to(ROOT)) for folder in ("cartpole", "cartpole_tuning")
          for path in (ROOT / "src" / folder).glob("*.py")),
    })
    manifest = {"score_version": 1, "scenarios": [s.snapshot() for s in scenarios],
                "validation_scenarios": [s.snapshot() for s in validation_scenarios],
                "seeds": args.seeds, "validation_seeds": args.validation_seeds,
                "bounds": dict(zip((*q_parameters(scenarios[0].plant.state_size), "r"), bounds)), "initial_q": initial_q, "initial_r": initial_r,
                "sampler_seed": args.sampler_seed, "startup_trials": args.startup_trials,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "scipy": scipy.__version__, "optuna": optuna_version},
                "sources": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}}
    # Normalize tuples before comparing with a manifest loaded from JSON.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    if args.resume:
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Resume settings/configs/code differ from manifest.json; use a new output directory.")
        if not (output / "study.sqlite3").is_file():
            raise ValueError("No study.sqlite3 found to resume.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "manifest.json", manifest)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        import optuna
    except ImportError:
        parser.error("Install optimization dependencies: .venv/bin/python -m pip install -e '.[optimize]'")
    try:
        validate_options(args)
        search = prepare_search(args)
        prepare_output(args, search, optuna.__version__)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{(search.output / 'study.sqlite3').resolve()}")
    try:
        return run_study(optuna, storage, args, search)
    finally:
        storage.remove_session()
        storage.engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
