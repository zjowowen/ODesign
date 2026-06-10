#!/usr/bin/env bash
set -eo pipefail

ROOT=${ROOT:-/mnt/shared-storage-user/ai4sreason/zhangjinouwen/Project/debug_5/ODesign}
RECORD_DIR=${RECORD_DIR:-"$ROOT/.cluster_operator/padding-batch-task5"}
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
  > "$RECORD_DIR/task5-e2e.stdout" \
  2> "$RECORD_DIR/task5-e2e.stderr"
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


test_path = Path("tests/test_padding_batch_end_to_end.py")
spec = importlib.util.spec_from_file_location(
    "test_padding_batch_end_to_end_h", test_path
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

tests = [
    "test_padding_batch_train_forward_backward_smoke",
]
results = []
for name in tests:
    try:
        getattr(mod, name)()
        results.append({"name": name, "status": "PASS"})
        print(f"PASS {name}")
    except Exception:
        tb = traceback.format_exc()
        results.append({"name": name, "status": "FAIL", "traceback": tb})
        print(f"FAIL {name}")
        print(tb)

failed = [result for result in results if result["status"] != "PASS"]
print(json.dumps({"results": results}, ensure_ascii=False))
raise SystemExit(1 if failed else 0)
PY_RUNNER
rc=$?
set -e

export TASK5_RC="$rc"
python - <<'PY_RESULT'
import json
import os
import pathlib
import time

record_dir = pathlib.Path(os.environ["RECORD_DIR"])
result = {
    "run_id": "padding-batch-task5-e2e-" + time.strftime("%Y%m%d-%H%M%S"),
    "ok": os.environ["TASK5_RC"] == "0",
    "returncode": int(os.environ["TASK5_RC"]),
    "command": "python importlib runner for tests/test_padding_batch_end_to_end.py",
    "stdout": (record_dir / "task5-e2e.stdout").read_text(errors="replace")[-20000:],
    "stderr": (record_dir / "task5-e2e.stderr").read_text(errors="replace")[-20000:],
}
(record_dir / "result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2)
)
print(json.dumps({k: v for k, v in result.items() if k not in ("stdout", "stderr")}, ensure_ascii=False))
PY_RESULT

cat "$RECORD_DIR/task5-e2e.stdout" || true
cat "$RECORD_DIR/task5-e2e.stderr" >&2 || true
exit "$rc"
