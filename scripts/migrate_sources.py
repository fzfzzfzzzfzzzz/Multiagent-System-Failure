"""Reuse source executions after a report-only protocol revision.

Source trajectories were completed before report generation. This tool copies
them to a fresh run root, strips every prior report, and records file hashes.
It never changes or selects source outcomes.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
from teamlearn.common import read_json,write_json,write_jsonl
from teamlearn.pipeline import code_hash


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);a=p.parse_args()
    old=Path(a.old).resolve();new=Path(a.new).resolve()
    if old==new or new.is_relative_to(old):raise ValueError('migration destination must be a separate directory')
    destination=new/'sources';destination.mkdir(parents=True,exist_ok=True)
    manifest=[];summaries=[]
    for source in sorted((old/'sources').glob('*.json')):
        record=read_json(source)
        if record.get('status')=='failure' and (not record.get('selected') or not record['selected']['score']['recurrence']):
            raise ValueError(f'invalid qualified source: {source.name}')
        old_code=record.get('code_hash')
        record.pop('experience',None)
        record['source_execution_code_hash']=old_code
        record['code_hash']=code_hash()
        record['report_protocol_migration']={'source_file_sha256':sha(source),'reason':'strict-v2 bounds and fixed retry; source execution precedes report generation'}
        target=destination/source.name
        if target.exists() and read_json(target)!=record:raise ValueError(f'migration target differs: {target}')
        write_json(target,record)
        manifest.append({'source':str(source),'source_sha256':sha(source),'target':str(target),'meta_id':record['meta_id'],'status':record['status']})
        summaries.append({k:v for k,v in record.items() if k not in ('selected','experience')})
    attempts_old=old/'source_attempts';attempts_new=new/'source_attempts';attempts_new.mkdir(parents=True,exist_ok=True)
    for source in sorted(attempts_old.glob('*.json')):
        target=attempts_new/source.name
        if target.exists() and sha(target)!=sha(source):raise ValueError(f'attempt target differs: {target}')
        if not target.exists():shutil.copy2(source,target)
    write_jsonl(new/'source_collection.jsonl',summaries)
    write_json(new/'migration_manifest.json',{'old':str(old),'new':str(new),'records':manifest,'source_attempt_files':len(list(attempts_new.glob('*.json'))),'new_code_hash':code_hash()})
    print({'records':len(manifest),'attempts':len(list(attempts_new.glob('*.json'))),'new':str(new)})


if __name__=='__main__':main()
