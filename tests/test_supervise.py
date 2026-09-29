import sys
from types import SimpleNamespace

sys.modules.setdefault("fcntl", SimpleNamespace())
from scripts.supervise import classify_gpu_processes


def test_cleanup_verification_distinguishes_foreign_gpu_processes():
    snapshot = [{"gpu": 4, "compute_pids": [20, 30]}, {"gpu": 5, "compute_pids": []}]
    owned, foreign = classify_gpu_processes(snapshot, {10, 20})
    assert owned == [20]
    assert foreign == [30]


def test_foreign_process_does_not_make_owned_cleanup_fail():
    owned, foreign = classify_gpu_processes([{"gpu": 4, "compute_pids": [30]}], {10, 20})
    assert not owned
    assert foreign == [30]
