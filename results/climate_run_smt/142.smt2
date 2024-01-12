(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsStandingNear (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (IsStandingNear a b))) (exists ((c BoundSet)) (exists ((a BoundSet)) (IsStandingNear a c))))))
(check-sat)
(get-model)