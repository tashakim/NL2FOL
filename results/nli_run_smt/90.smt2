(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsKickboxing (BoundSet) Bool)
(declare-fun IsTwoPeople (BoundSet) Bool)
(declare-fun IsTicking (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (IsKickboxing a) (IsTwoPeople a))) (exists ((a BoundSet)) (exists ((b BoundSet)) (IsTicking a b))))))
(check-sat)
(get-model)