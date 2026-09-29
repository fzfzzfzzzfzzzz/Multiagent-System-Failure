from teamlearn.datasets import generate
from teamlearn.pipeline import collect
from teamlearn.stream import run_stream
from teamlearn.common import read_jsonl


def test_online_same_library_and_read_only_probes(tmp_path):
    dataset=tmp_path/"data";source=tmp_path/"sources";out=tmp_path/"stream"
    generate(dataset,1)
    for split in ("train","dev"):
        collect(dataset,source,split,"configs/models.yaml","mock",limit=6,seeds=(11,),rounds=1)
    run_stream(dataset,source,out,"configs/models.yaml","mock",steps=2,streams=1,probe_every=1,rounds=1)
    rows=read_jsonl(out/"results.jsonl")
    for step in range(2):
        group={r["group"]:r for r in rows if r["step"]==step}
        assert group["L1"]["memory_hash"]==group["L2"]["memory_hash"]
    assert len(read_jsonl(out/"probes.jsonl"))==24
