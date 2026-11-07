"""Automatic reward boundaries reuse the verified same-run restoration path."""
from brass_ai.bank import read
from brass_ai.learning.contracts import require
from brass_ai.step5.league import implementation_id
from . import entropy_schedule as S


def run(root,run_path,config,*,runner,iterations,**kwargs):
    """Called under the run lock. Close workers before restoring a new stage.

    Full checkpoint restoration retains both Adam states and RNG streams. The
    existing durable source transition retires incompatible PPG targets before
    collection; an interruption at any point replays that same transition.
    """
    path=root/run_path/'state.json';remaining=iterations;summaries=[]
    require(type(iterations) is int and iterations>0,'Iterations must be positive')
    while remaining:
        state=read(path) if path.exists() else None
        start=state['iteration'] if state else 0
        if state is not None and S.automatic(config):
            config=S.follow_committed(config,state)
            from .resume import check_signature
            signature=dict(state['signature'],config=config,implementation=implementation_id())
            from brass_ai.rust_backend.lab_replay_compat import training_signature
            signature=training_signature(state,signature)
            check_signature(state,signature,reviewed=kwargs.get('resume_reviewed_source',False))
            target=S.automatic_next(config,state)
            if target!=config:
                receipt=S.automatic_receipt(target,state)
                print(f'Automatic insolvency stage at g{start}, k={receipt["progress"]}: '
                    f'{receipt["before"]} -> {receipt["after"]}; fresh reward-compatible PPG cache',flush=True)
                config=target
                # Permission comes from the validated, already opted-in plan;
                # the exact migration still passes the pinned source review.
                kwargs['resume_reviewed_source']=True
        result=runner(root,run_path,config,iterations=remaining,**kwargs)
        done=result['completed_iteration']-start
        require(0<done<=remaining,'Automatic stage runner made invalid committed progress')
        remaining-=done;summaries=(summaries+result.get('summaries',[]))[-2:]
        if remaining:
            state=read(path)
            require(S.automatic(config) and S.waiting(S.restore_run(config,state)),
                'Training returned early without an automatic reward boundary')
    result['summaries']=summaries
    return result
