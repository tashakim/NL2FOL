(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInRunningClothes (BoundSet) Bool)
(declare-fun IsStretching (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsInRunningClothes a) (IsStretching c)))) (exists ((d BoundSet)) (IsStretching d)))))
(check-sat)
(get-model)