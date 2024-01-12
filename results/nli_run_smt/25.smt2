(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsDriving (BoundSet) Bool)
(declare-fun IsNear (BoundSet) Bool)
(declare-fun IsAt (BoundSet) Bool)
(declare-fun IsNotAt (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsDriving a) (and (IsNear b) (IsAt c)))))) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (IsNotAt a) (IsAt d)))))))
(check-sat)
(get-model)