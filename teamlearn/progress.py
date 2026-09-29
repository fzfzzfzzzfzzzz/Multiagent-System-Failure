"""External watchdog signal; only real work, never an unconditional timer."""
import json
import os
from pathlib import Path
import time


def progress(event,**fields):
    path=os.getenv('TEAMLEARN_HEARTBEAT')
    if path:
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix(f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps({'at':time.time(),'pid':os.getpid(),'event':event,**fields}))
        temp.replace(target)
