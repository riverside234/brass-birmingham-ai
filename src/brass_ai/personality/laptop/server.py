"""Local stdin/stdout service. No sockets; one admitted game job at a time."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from threading import Lock,Event

from brass_ai.learning.contracts import require
from .contracts import PROTOCOL
from .package import validate
from .runtime import Runtime
from .sampling import new_save,restore


def public_error(error):
    # Schema/native exceptions can include the rejected observation and own
    # private cards. The game popup must never receive their string payload.
    return type(error).__name__+': neural request failed validation or execution; no move published'


def serve(directory,backend='ort-cpu',*,fixture=False,mono=None):
    package=validate(directory,allow_fixture=fixture);runtime=None;pending=None;active_id=None
    output=Lock();stop=Event();cancel=Event();pool=ThreadPoolExecutor(max_workers=1)
    def emit(value):
        with output:sys.stdout.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n');sys.stdout.flush()
    def execute(request):
        nonlocal runtime
        try:
            if runtime is None:runtime=Runtime(directory,backend,fixture=fixture,mono=mono)
            if stop.is_set():runtime.interrupt(cancel=cancel.is_set())
            result=runtime.plan(request['body'])
            return dict(protocol=PROTOCOL,id=request['id'],ok=True,data=result)
        except Exception as error:
            # No private observations in diagnostics/UI.
            return dict(protocol=PROTOCOL,id=request['id'],ok=False,error=public_error(error))
    try:
        for line in sys.stdin:
            request={}
            try:
                require(len(line)<=8*1024*1024,'Oversized game request')
                request=json.loads(line);require(request.get('protocol')==PROTOCOL and isinstance(request.get('id'),str),'Invalid service envelope')
                op=request.get('op')
                if op in ('cancel','play-now'):
                    # Late UI controls cannot interrupt the next turn or leave
                    # an unsolicited error response on its private reply pipe.
                    if request['id']!=active_id or pending is None or pending.done():continue
                    stop.set()
                    if op=='cancel':cancel.set()
                    if runtime is not None:runtime.interrupt(cancel=cancel.is_set())
                    continue  # The turn response comes only after full retirement.
                require(pending is None or pending.done(),'A turn is already running')
                if op=='plan':
                    stop.clear();cancel.clear()
                    if runtime is not None:runtime.interrupted.clear();runtime.cancelled.clear()
                    active_id=request['id'];pending=pool.submit(execute,request)
                    # Mark the future completed BEFORE publishing its reply. A
                    # save/restore immediately after a reply must not see "busy".
                    pending.add_done_callback(lambda f:emit(f.result()))
                    continue
                if op=='initialize':data=new_save(package,**request['body'])
                elif op=='restore':data=restore(package,**request['body'])
                elif op=='hello':data=dict(package_id=package['id'],fixture=package['fixture'],backend=backend,workers=2)
                elif op=='close':break
                else:raise ValueError('Unknown game operation')
                emit(dict(protocol=PROTOCOL,id=request['id'],ok=True,data=data))
            except Exception as error:emit(dict(protocol=PROTOCOL,id=request.get('id'),ok=False,error=public_error(error)))
    finally:
        stop.set();cancel.set()
        if runtime is not None:runtime.interrupt(cancel=True)
        pool.shutdown(wait=True)
        if runtime is not None:runtime.close()
