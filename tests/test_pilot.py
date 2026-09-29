import json
from pathlib import Path
from scripts.pilot_job import select_candidates
from teamlearn.progress import progress


def test_pilot_selection_balanced_and_outcome_free():
    rows=[]
    for mechanism in ('F1','F2','F3','F4'):
        for family in ('a','b','c','d','e','f'):
            for n in range(3):rows.append({'meta_id':f'{mechanism}-{family}-{n}','mechanism':mechanism,'family_id':family,'backend':'workflow'})
    selected=select_candidates(rows)
    assert len(selected)==32
    assert selected==select_candidates(list(reversed(rows)))
    for mechanism in ('F1','F2','F3','F4'):
        group=[r for r in selected if r['mechanism']==mechanism]
        assert len(group)==8 and len({r['family_id'] for r in group})==6


def test_progress_is_opt_in_and_only_explicit_work(tmp_path,monkeypatch):
    target=tmp_path/'heartbeat.json'
    monkeypatch.delenv('TEAMLEARN_HEARTBEAT',raising=False)
    progress('noop');assert not target.exists()
    monkeypatch.setenv('TEAMLEARN_HEARTBEAT',str(target))
    progress('completed',output_tokens=2)
    assert json.loads(target.read_text())['output_tokens']==2
