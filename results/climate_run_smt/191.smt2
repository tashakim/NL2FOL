(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInBlueShirt (BoundSet) Bool)
(declare-fun IsHoldingSnowball (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (IsInBlueShirt a) (IsHoldingSnowball a))) (exists ((d BoundSet)) (IsHoldingSnowball d)))))
(check-sat)
(get-model)