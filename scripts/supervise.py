"""Linux experiment owner: inference and client share a systemd control group.

The user service's KillMode=control-group handles even supervisor SIGKILL. This
process additionally detects idle/stuck clients and kills only its own groups.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request


def save(path, value):
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(target)


def stop_group(process, grace=15):
    if process is None:return
    # Children can survive a crashed group leader. Never skip killpg merely
    # because Popen.poll() says that the leader has exited.
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    deadline=time.monotonic()+grace
    while time.monotonic()<deadline:
        process.poll()
        try:os.killpg(process.pid,0)
        except ProcessLookupError:return
        time.sleep(.2)
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:pass


def gpu_snapshot(gpus):
    if not gpus:return []
    import pynvml as nv
    nv.nvmlInit();result=[]
    try:
        for gpu in gpus:
            handle=nv.nvmlDeviceGetHandleByIndex(gpu)
            memory=nv.nvmlDeviceGetMemoryInfo(handle).used//1048576
            pids=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)]
            util=nv.nvmlDeviceGetUtilizationRates(handle).gpu
            result.append({'gpu':gpu,'used_mb':memory,'util':util,'compute_pids':pids})
    finally:nv.nvmlShutdown()
    return result


def check_gpus(gpus,max_used_mb=512):
    result=gpu_snapshot(gpus)
    for row in result:
        memory=row['used_mb'];pids=row['compute_pids'];util=row['util'];gpu=row['gpu']
        if memory>max_used_mb or pids or util>5:
            raise RuntimeError(f'GPU {gpu} is occupied: {row}')
    return result


def descendants(pid):
    """Snapshot a process tree from /proc before cleanup."""
    found=set();pending=[pid] if pid else []
    while pending:
        parent=pending.pop()
        if parent in found:continue
        found.add(parent)
        children=Path(f'/proc/{parent}/task/{parent}/children')
        try:pending.extend(int(value) for value in children.read_text().split())
        except (OSError,ValueError):pass
    return found


def classify_gpu_processes(snapshot,owned_pids):
    active={pid for row in snapshot for pid in row['compute_pids']}
    return sorted(active & set(owned_pids)),sorted(active-set(owned_pids))


def supervise(config):
    state=Path(config['state_dir']).resolve();state.mkdir(parents=True,exist_ok=True)
    status={'started_at':time.time(),'supervisor_pid':os.getpid(),'state':'preflight','config':config}
    save(state/'status.json',status)
    # flock is host-wide per GPU for this project; NVML guards other projects.
    locks=[];server=None;worker=None;logs=[]
    lock_dir=Path('/data/fangc/teamlearn/locks') if config.get('gpus') else state/'locks'
    lock_dir.mkdir(parents=True,exist_ok=True)
    interrupted=[]
    def on_signal(signum,frame):
        interrupted.append(signum)
        raise InterruptedError(f'signal {signum}')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,on_signal)
    started=time.monotonic();heartbeat=state/'client_heartbeat.json'
    try:
        for gpu in sorted(config.get('gpus',[])):
            handle=(lock_dir/f'gpu-{gpu}.lock').open('a+')
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        status['gpu_before']=check_gpus(config.get('gpus',[]))
        if shutil.disk_usage(state).free<config.get('minimum_disk_gb',5)*1024**3:raise RuntimeError('insufficient disk space')
        if config.get('port'):
            with socket.socket() as check:
                check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                check.bind(('127.0.0.1',config['port']))
        env={**os.environ,**config.get('env',{}),'TEAMLEARN_HEARTBEAT':str(heartbeat),'PYTHONUNBUFFERED':'1'}
        server_log=(state/'model.log').open('a');logs.append(server_log)
        server=subprocess.Popen(config['server_command'],cwd=config['cwd'],env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(state='starting_model',server_pid=server.pid);save(state/'status.json',status)
        ready_deadline=time.monotonic()+config.get('startup_timeout',600)
        while True:
            if server.poll() is not None:raise RuntimeError(f'model exited during startup: {server.returncode}')
            try:
                with urllib.request.urlopen(config['ready_url'],timeout=3) as response:
                    payload=json.load(response)
                if config.get('expected_model') and config['expected_model'] not in [m['id'] for m in payload.get('data',[])]:
                    raise RuntimeError('ready endpoint returned an unexpected model')
                break
            except (OSError,ValueError):
                if time.monotonic()>ready_deadline:raise TimeoutError('model startup timeout')
                time.sleep(2)
        save(heartbeat,{'at':time.time(),'event':'client_starting'})
        client_log=(state/'experiment.log').open('a');logs.append(client_log)
        worker=subprocess.Popen(config['client_command'],cwd=config['cwd'],env=env,stdout=client_log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(state='running',client_pid=worker.pid,model_ready_at=time.time());save(state/'status.json',status)
        while worker.poll() is None:
            if server.poll() is not None:raise RuntimeError(f'model exited unexpectedly: {server.returncode}')
            idle=time.time()-heartbeat.stat().st_mtime
            if idle>config.get('idle_timeout',300):raise TimeoutError(f'no completed model call/client progress for {idle:.0f}s')
            if time.monotonic()-started>config.get('max_runtime',43200):raise TimeoutError('bounded run time exhausted; resume from checkpoints')
            if shutil.disk_usage(state).free<config.get('minimum_disk_gb',5)*1024**3:raise RuntimeError('disk free-space floor reached')
            status.update(checked_at=time.time(),heartbeat_age_seconds=round(idle,2))
            save(state/'status.json',status);time.sleep(config.get('poll_seconds',5))
        if worker.returncode:raise RuntimeError(f'experiment exited with code {worker.returncode}; inspect experiment.log')
        status.update(state='completed',client_exit_code=0)
    except BaseException as error:
        status.update(state='failed',error=f'{type(error).__name__}: {error}')
    finally:
        # Finish cleanup even if another stop signal arrives mid-cleanup.
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
        owned_pids=descendants(worker.pid if worker else None)|descendants(server.pid if server else None)
        stop_group(worker,config.get('cleanup_grace',15))
        stop_group(server,config.get('cleanup_grace',15))
        for handle in logs:handle.close()
        status.update(finished_at=time.time(),cleanup_requested=True)
        try:
            status['gpu_after']=gpu_snapshot(config.get('gpus',[]))
            remaining,foreign=classify_gpu_processes(status['gpu_after'],owned_pids)
            status['owned_gpu_pids_remaining']=remaining
            status['foreign_gpu_pids_after']=foreign
            status['gpu_released']=not remaining
            if remaining:status['cleanup_verification_error']=f'owned GPU processes remain: {remaining}'
        except Exception as error:
            status['gpu_released']=False;status['cleanup_verification_error']=str(error)
        save(state/'status.json',status)
        for handle in locks:handle.close()
    return 0 if status['state']=='completed' and status.get('gpu_released') else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();raise SystemExit(supervise(json.loads(Path(args.config).read_text())))
