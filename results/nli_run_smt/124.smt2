(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsWalking (BoundSet) Bool)
(declare-fun IsInGrass (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsWalking a)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsWalking a) (IsInGrass b)))))))
(check-sat)
(get-model)