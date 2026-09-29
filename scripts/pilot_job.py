"""Balanced E0 entry run; freeze candidate IDs before reading any outcomes."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
from teamlearn.common import read_json,read_jsonl,write_json,digest
from teamlearn.pipeline import collect,source_records,run_experiment
from teamlearn.progress import progress


def select_candidates(rows,per_mechanism=8):
    selected=[]
    for mechanism in ('F1','F2','F3','F4'):
        families=defaultdict(list)
        for row in rows:
            if row['mechanism']==mechanism and row['backend']=='workflow':families[row['family_id']].append(row)
        groups=[sorted(values,key=lambda x:x['meta_id']) for _,values in sorted(families.items())]
        ordered=[group[i] for i in range(max(map(len,groups),default=0)) for group in groups if i<len(group)]
        if len(ordered)<per_mechanism:raise ValueError('not enough candidates')
        selected.extend(ordered[:per_mechanism])
    return selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='datasets');p.add_argument('--out',required=True)
    p.add_argument('--models',default='configs/models.yaml');p.add_argument('--model',default='qwen3_8b')
    p.add_argument('--turns',type=int,default=7);a=p.parse_args()
    out=Path(a.out);selected=select_candidates(read_jsonl(Path(a.dataset)/'dev_manifest.jsonl'))
    design={'stage':'E0-development-only','dataset_hash':read_json(Path(a.dataset)/'manifest.json')['hash'],
            'selection':'8 candidates per F1-F4, round-robin across development rule families; no outcome selection',
            'meta_ids':[r['meta_id'] for r in selected],'source_seeds':[11,29,47],'target_seeds':[101,211],
            'rounds':2,'turns_per_role':a.turns,'model':a.model,'arms':['no_memory','uniform','source_evidence','full_history']}
    if (out/'design.json').exists() and read_json(out/'design.json')!=design:raise ValueError('pilot design changed; new directory required')
    write_json(out/'design.json',design);progress('pilot_design_frozen',n_candidates=len(selected))
    collect(a.dataset,out/'collected','dev',a.models,a.model,seeds=(11,29,47),rounds=2,turns=a.turns,meta_ids=set(design['meta_ids']))
    records=source_records(out/'collected','dev')
    counts={m:sum(r['mechanism']==m for r in records) for m in ('F1','F2','F3','F4')}
    write_json(out/'gate.json',{'counts':counts,'minimum_per_mechanism':1,'passed':all(counts.values())})
    if not all(counts.values()):raise RuntimeError(f'E0 coverage gate failed: {counts}; stop and release GPU without altering test tasks')
    progress('starting_E0',counts=counts)
    result=run_experiment(a.dataset,out/'collected',out/'E0','E0','dev',a.models,a.model,seeds=(101,211),rounds=2,turns=a.turns)
    write_json(out/'completion.json',result);progress('pilot_completed')


if __name__=='__main__':main()
