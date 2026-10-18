"""MAP Plackett-Luce ratings with paired setup-cluster bootstrap intervals."""
import math
import numpy as np
from brass_ai.learning.contracts import require

IL='generation-0'
V4='v4'


def check_dependencies():
    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise ValueError('Strength ratings need scipy; install requirements-step5.txt before training') from error
    return minimize


def encode(games,agents):
    agents=[IL]+sorted(set(agents)-{IL});index={a:i for i,a in enumerate(agents)}
    games=sorted(games,key=lambda g:(g['slot'],g['setup_id'],g['candidate_seat'],g['game_id']))
    require(games and len({g['game_id'] for g in games})==len(games),'Missing/duplicate rated games')
    orders=np.asarray([g['order'] for g in games],dtype=np.int64)
    require(orders.shape==(len(games),4) and np.all(np.sort(orders,axis=1)==np.arange(4)),
            'Rating order must be a permutation of four seats')
    seats=np.asarray([[index[a] for a in g['agents']] for g in games],dtype=np.int64)
    require(seats.shape==orders.shape,'Rating needs four seat agents')
    ordered=np.take_along_axis(seats,orders,axis=1)
    return games,agents,ordered,orders


def objective(x,ordered,orders,weights,agent_count,theta_sd,seat_sd):
    theta=np.r_[0.,x[:agent_count-1]]
    s=np.r_[x[agent_count-1:],-x[agent_count-1:].sum()]
    utility=theta[ordered]+s[orders];du=np.zeros_like(utility)
    value=0.
    for k in range(3):
        tail=utility[:,k:];top=tail.max(axis=1);exp=np.exp(tail-top[:,None]);total=exp.sum(axis=1)
        value+=float(np.dot(weights,top+np.log(total)-utility[:,k]))
        du[:,k:]+=weights[:,None]*(exp/total[:,None]);du[:,k]-=weights
    gt=np.zeros(agent_count);gs=np.zeros(4)
    np.add.at(gt,ordered.ravel(),du.ravel());np.add.at(gs,orders.ravel(),du.ravel())
    value+=float(np.dot(theta,theta)/(2*theta_sd**2)+np.dot(s,s)/(2*seat_sd**2))
    gt+=theta/theta_sd**2;gs+=s/seat_sd**2
    return value,np.r_[gt[1:],gs[:3]-gs[3]]


def fit_encoded(ordered,orders,agent_count,config,weights=None):
    minimize=check_dependencies();weights=np.ones(len(ordered)) if weights is None else np.asarray(weights,dtype=np.float64)
    require(weights.shape==(len(ordered),) and np.isfinite(weights).all() and (weights>=0).all() and weights.sum()>0,
            'Invalid bootstrap game weights')
    result=minimize(objective,np.zeros(agent_count+2),
        args=(ordered,orders,weights,agent_count,config['theta_prior_sd'],config['seat_prior_sd']),
        method='L-BFGS-B',jac=True,options=dict(maxiter=1000,ftol=1e-12,gtol=1e-7,maxls=50))
    gradient=float(np.max(np.abs(result.jac)))
    require(np.isfinite(result.fun) and np.isfinite(result.x).all() and
            (result.success or gradient<1e-5),'Plackett-Luce fit did not converge: '+str(result.message))
    theta=np.r_[0.,result.x[:agent_count-1]];seat=np.r_[result.x[agent_count-1:],-result.x[agent_count-1:].sum()]
    return theta,seat,dict(converged=True,iterations=int(result.nit),objective=float(result.fun),gradient_max=gradient)


def fit(games,agents,config):
    games,agents,ordered,orders=encode(games,agents)
    theta,seat,diagnostic=fit_encoded(ordered,orders,len(agents),config)
    return dict(agents=agents,theta=theta.tolist(),seat=seat.tolist(),diagnostic=diagnostic)


def estimate(games,agents,config):
    games,agents,ordered,orders=encode(games,agents);n=len(agents)
    theta,seat,diagnostic=fit_encoded(ordered,orders,n,config)
    # A setup gets one bootstrap multiplicity across ALL slots and rotations.
    setups=sorted({g['setup_id'] for g in games});lookup={s:i for i,s in enumerate(setups)}
    cluster=np.asarray([lookup[g['setup_id']] for g in games]);rng=np.random.default_rng(config['seed'])
    samples=[]
    for _ in range(config['bootstrap']):
        counts=np.bincount(rng.integers(len(setups),size=len(setups)),minlength=len(setups))
        draw,_,_=fit_encoded(ordered,orders,n,config,weights=counts[cluster]);samples.append(draw)
    samples=np.asarray(samples);low,high=np.quantile(samples,[.025,.975],axis=0)
    scale=config['elo_scale'];v4=agents.index(V4)
    sigmoid=lambda x:float(1/(1+math.exp(-max(-700.,min(700.,float(x))))))
    rated={a:dict(theta=float(theta[i]),points=scale*float(theta[i]),theta_ci95=[float(low[i]),float(high[i])],
        points_ci95=[scale*float(low[i]),scale*float(high[i])],
        probability_above_IL=sigmoid(theta[i]),probability_above_v4=sigmoid(theta[i]-theta[v4])) for i,a in enumerate(agents)}
    differences=[]
    for i,a in enumerate(agents):
        for j in range(i+1,n):
            interval=np.quantile(samples[:,j]-samples[:,i],[.025,.975])*scale
            differences.append(dict(a=a,b=agents[j],points_delta=scale*float(theta[j]-theta[i]),
                points_delta_ci95=interval.tolist(),b_stronger=bool(interval[0]>0),a_stronger=bool(interval[1]<0)))
    return dict(contract=config['contract'],ratings=rated,seat_offsets=seat.tolist(),fit=diagnostic,
        games=len(games),setups=len(setups),bootstrap_refits=len(samples),differences=differences,
        scope='Held-out mixed-table PL model with Gaussian priors; IL theta fixed at zero; setup-cluster percentile intervals; seat-neutral outrank probabilities'),samples
