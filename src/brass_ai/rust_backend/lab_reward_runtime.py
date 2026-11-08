"""Additive teacher-only support for zero charge and canal/final VP timing.

Patch a detached, sealed teacher core; the installed game and ordinary binary
remain unchanged. Positive-cost search equations and traversal are unchanged.
"""
import hashlib
from pathlib import Path

from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from .lab_contract import native_fingerprint,pin

CONTRACT='brass-zero-insolvency-teacher/1.0.0'
SELECTOR='artifacts/rust-backend/teacher-zero-insolvency/build.json'
BASE_CORE='8d59ec4b5716c3b16c9d00d97975f589e481b629afc85f1732db4e37549db4cf'
ZERO_CORE='e6bedd50fa304f7bc69c7bd4916b75050973760d01205711fb9386edb1518c52'
OLD='                && c.event_cost > 0.\n'
NEW='                && c.event_cost >= 0.\n                && c.max_cost > 0.\n'
OLD_VERSION='(self.version == "brass-combined-return/2.0.0" && self.vp_cap == Some(180))'
NEW_VERSION='(matches!(self.version.as_str(), "brass-combined-return/2.0.0" | "brass-combined-return/3.0.0") && self.vp_cap == Some(180))'
TESTS='''
#[cfg(test)]
mod post_transfer_reward_tests {
    use super::*;
    fn reward(cost: f64) -> Reward {
        Reward { version: "brass-combined-return/3.0.0".into(), win: 1., placement: 0.05,
            vp: 0.001, vp_cap: Some(180), insolvency: Insolvency { event_cost: cost, max_cost: 0.025 } }
    }
    #[test]
    fn post_transfer_zero_charge_and_placement() {
        let r = reward(0.); r.validate().unwrap();
        let mut ledger = Ledger::new(r.clone(), LedgerState { spent: [0.;4], settled: vec![] }).unwrap();
        for _ in 0..100 { for seat in 0..4 { ledger.charge(seat); } }
        assert_eq!(ledger.spent, [0.;4]); assert_eq!(ledger.remaining(), [0.025;4]);
        let seats: Vec<_> = (0..4).map(|s| serde_json::json!({"seat":s,"vp":0,"income":0,
            "cash":4-s,"rank":s+1,"utility":if s==0 {1.} else {0.}})).collect();
        let result: ResultDto = serde_json::from_value(serde_json::json!({"objective":"win-share/1.0.0","seats":seats})).unwrap();
        let u = r.terminal(&result).unwrap();
        for (a,b) in u.into_iter().zip([1.05,0.05*2./3.,0.05/3.,0.]) { assert!((a-b).abs()<1e-14); }
    }
    #[test]
    fn post_transfer_positive_cost_is_preserved() {
        let mut r=reward(0.0025);r.placement=0.1;r.version="brass-combined-return/2.0.0".into();r.validate().unwrap();
        let mut ledger=Ledger::new(r,LedgerState { spent:[0.;4],settled:vec![] }).unwrap();
        ledger.charge(0);ledger.charge(0);assert_eq!(ledger.spent,[0.005,0.,0.,0.]);
        for _ in 0..100 { ledger.charge(0); } assert!((ledger.spent[0]-0.025).abs()<1e-14);
    }
    #[test]
    fn post_transfer_invalid_budgets_rejected() {
        for cost in [-0.001,f64::NAN,f64::INFINITY] { assert!(reward(cost).validate().is_err()); }
        let mut r=reward(0.);r.insolvency.max_cost=0.;assert!(r.validate().is_err());
    }
    #[test]
    fn post_transfer_vp_payments_cap_once_and_preserve_total() {
        let r=reward(0.);r.validate().unwrap();
        let pay=|vp:f64| r.vp * vp.min(f64::from(r.vp_cap.unwrap()));
        for (canal,final_vp) in [(60.,100.),(90.,200.),(180.,200.),(200.,220.)] {
            let first=pay(canal);let last=pay(final_vp)-first;
            assert!((first+last-pay(final_vp)).abs()<1e-14);
        }
        // The raw search value is full-game utility. The already paid canal
        // credit is a constant offset; selection/backup equations stay intact.
        assert!((pay(60.)-0.06).abs()<1e-14);assert!((pay(200.)-pay(60.)-0.12).abs()<1e-14);
    }
}
'''


def fingerprint():return digest(dict(contract=CONTRACT,old=OLD,new=NEW,old_version=OLD_VERSION,new_version=NEW_VERSION,tests=TESTS,base=BASE_CORE,core=ZERO_CORE))


def patch(directory):
    path=Path(directory)/'brass-ai-core/src/search/reward.rs'
    text=path.read_text(encoding='utf-8')
    require(text.count(OLD)==1 and text.count(OLD_VERSION)==1 and 'post_transfer_reward_tests' not in text,'Zero-reward core patch drift')
    path.write_bytes((text.replace(OLD,NEW).replace(OLD_VERSION,NEW_VERSION)+TESTS).encode())


def apply(directory):
    require(native_fingerprint(directory)==BASE_CORE,'Zero-reward patch requires the exact sealed teacher core')
    patch(directory)
    require(native_fingerprint(directory)==ZERO_CORE,'Patched reward core identity differs')
    return dict(contract=CONTRACT,source=fingerprint(),base_core_source=BASE_CORE,core_source=ZERO_CORE)


def resolve(root,original):
    from .lab_paths import same_pin
    from .lab_backend import checked_pin
    from .lab_replay_compat import accepts,reviewed_validator
    path=Path(root)/SELECTOR
    require(path.is_file(),'Prepare the zero-insolvency teacher with scripts/prepare-rust-post-transfer-reward.py')
    receipt=pin(path);value=checked_pin(receipt)
    require(value.get('contract')=='brass-lab-b07-teacher-build/1.0.0' and
        value.get('reward_support')==dict(contract=CONTRACT,source=fingerprint(),base_core_source=BASE_CORE,core_source=ZERO_CORE)
        and original['core_source']==BASE_CORE and value['core_source']==ZERO_CORE
        and same_pin(value['original_verification_receipt'],original['receipt'])
        and value['producer']==original['producer'] and value['bootstrap']==original['bootstrap']
        and (accepts(value['implementation']) or reviewed_validator(value['implementation']))
        and value.get('wire_regressions_passed') is True,'Zero-insolvency teacher source/parent differs')
    for key in ('binary','tests','build_log','reward_tests'):
        item=value[key];require(pin(item['path'])==item,'Zero-insolvency teacher evidence changed')
    return dict(value,receipt=receipt)
