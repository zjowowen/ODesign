#!/usr/bin/env bash
set -eo pipefail

ROOT=${ROOT:-/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign}
RECORD_DIR="$ROOT/.cluster_operator/padding-batch-task4"
mkdir -p "$RECORD_DIR"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate odesign || true
fi

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/src:${PYTHONPATH:-}"

set +e
python - <<'PY_RUNNER' \
  > "$RECORD_DIR/task4-pytest.stdout" \
  2> "$RECORD_DIR/task4-pytest.stderr"
import importlib.util
import re
import sys
import traceback
import types
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def raises(expected_exception, match=None):
    try:
        yield
    except expected_exception as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(
                f"exception message {exc!r} did not match {match!r}"
            ) from exc
        return
    raise AssertionError(f"did not raise {expected_exception!r}")


sys.modules.setdefault("pytest", types.SimpleNamespace(raises=raises))

test_path = Path("tests/test_padding_batch_training_contract.py")
spec = importlib.util.spec_from_file_location(
    "test_padding_batch_training_contract_h", test_path
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

failures = []
ran = []
for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    try:
        getattr(mod, name)()
        ran.append(name)
    except Exception:
        failures.append((name, traceback.format_exc()))

print(f"ran {len(ran) + len(failures)} contract tests")
for name in ran:
    print(f"PASS {name}")
for name, tb in failures:
    print(f"FAIL {name}")
    print(tb)

raise SystemExit(1 if failures else 0)
PY_RUNNER
rc=$?
set -e

export TASK4_RC="$rc"
python - <<'PY_RESULT'
import json
import os
import pathlib
import time

record_dir = pathlib.Path(
    "/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign/.cluster_operator/padding-batch-task4"
)
result = {
    "run_id": "padding-batch-task4-pytest-" + time.strftime("%Y%m%d-%H%M%S"),
    "ok": os.environ["TASK4_RC"] == "0",
    "returncode": int(os.environ["TASK4_RC"]),
    "command": "python importlib runner for tests/test_padding_batch_training_contract.py",
    "stdout": (record_dir / "task4-pytest.stdout").read_text(errors="replace")[-12000:],
    "stderr": (record_dir / "task4-pytest.stderr").read_text(errors="replace")[-12000:],
}
(record_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("stdout", "stderr")}, ensure_ascii=False))
PY_RESULT

cat "$RECORD_DIR/task4-pytest.stdout" || true
cat "$RECORD_DIR/task4-pytest.stderr" >&2 || true
exit "$rc"
