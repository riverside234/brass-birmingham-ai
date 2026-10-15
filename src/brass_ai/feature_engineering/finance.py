"""FE1: signed next-settlement exposure, provisional order and UI opportunities."""
from .contracts import Measure,KINDS
from .rules import has_settlement


def financial(view,cash=None,markers=None,debts=None):
    from .rules import income_level
    out={};observer=view['seat'];applicable=has_settlement(view['context'])
    for relative in range(4):
        seat=(observer+relative)%4;p=view['players'][seat]
        out[f'finance.{relative}.settlement']=Measure.exact(applicable)
        debt=p['debt'] if debts is None else debts[seat]
        income=p['income'] if markers is None else income_level(markers[seat])
        out[f'finance.{relative}.public_debt']=Measure.exact(debt)
        if not applicable:values={k:Measure.absent() for k in ('negative_payment','projected_cash','shortfall')}
        elif view['context']['phase']=='debt' and debt>0:
            values=dict(negative_payment=Measure.exact(max(0,-income)),projected_cash=Measure(),shortfall=Measure())
        else:
            balance=p['cash'] if cash is None else cash[seat]
            projection=balance+income
            values=dict(negative_payment=Measure.exact(max(0,-income)),projected_cash=Measure.exact(projection),shortfall=Measure.exact(max(0,-projection)))
        out.update({f'finance.{relative}.{k}':v for k,v in values.items()})
    return out


def provisional_order(view):
    observer=view['seat'];order=sorted(view['order'],key=lambda seat:view['players'][seat]['spent'])
    applicable=not(view['context']['terminal'] or view['context']['era']=='RailEra' and view['context']['round']==8)
    out={}
    for relative in range(4):
        seat=(observer+relative)%4
        out[f'order.{relative}.position']=Measure.exact(order.index(seat)) if applicable else Measure.absent()
        out[f'order.{relative}.spending_difference']=Measure.exact(view['players'][observer]['spent']-view['players'][seat]['spent'])
    return out


def opportunity_key(choice):
    kind=choice['kind']
    if kind=='scout':return (kind,tuple(sorted([choice['card']]+choice['cards'])))
    if kind=='build':return (kind,choice['target'],choice['industry'])
    if kind in ('develop','bonus_develop'):return (kind,choice['industry'])
    if kind=='sell':return (kind,choice['target'],choice['merchant'])
    if kind in ('coal','iron','beer','network','liquidate'):return (kind,choice['target'])
    return (kind,)


def opportunities(view,choices):
    phase=view['context']['phase']
    relevant={'action':{'build','develop','sell','loan','pass','scout','network'},
        'payment':{'coal','iron','beer'},'bonus':{'bonus_skip','bonus_develop'},
        'continuation':{'finish',view['pending']['kind']},'debt':{'liquidate'}}.get(phase,set())
    out={'opportunities.main_action_available':Measure.exact(phase=='action')}
    for kind in KINDS:
        keys={opportunity_key(c) for c in choices if c['kind']==kind}
        out[f'opportunities.{kind}.available']=Measure.exact(bool(keys)) if kind in relevant else Measure.absent()
        out[f'opportunities.{kind}.targets']=Measure.exact(len(keys)) if kind in relevant else Measure.absent()
    return out
