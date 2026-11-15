#!/usr/bin/env bash
# Remove generated outputs and caches, preserving source, config, and .venv.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

remove() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    rm -rf -- "$path"
    printf 'Removed: %s\n' "$path"
  fi
}

for path in results build dist __pycache__ .pytest_cache .mypy_cache .ruff_cache .mplconfig *.egg-info; do
  remove "$path"
done

for directory in src tests scripts; do
  if [[ -d "$directory" && ! -L "$directory" ]]; then
    while IFS= read -r -d '' path; do
      remove "$path"
    done < <(find "$directory" -type d \( -name __pycache__ -o -name '*.egg-info' \) -prune -print0)
  fi
done
