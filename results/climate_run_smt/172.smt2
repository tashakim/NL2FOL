(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsDigging (BoundSet) Bool)
(declare-fun IsOnBeach (BoundSet) Bool)
(declare-fun IsWearing (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsDigging a) (IsOnBeach b)))) (exists ((d BoundSet)) (exists ((a BoundSet)) (and (IsWearing d) (IsDigging a)))))))
(check-sat)
(get-model)