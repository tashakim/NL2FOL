(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsRunning (BoundSet) Bool)
(declare-fun IsJumping (BoundSet) Bool)
(declare-fun IsOutdoors (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsRunning a) (IsJumping a))) (and (forall ((d BoundSet)) (=> (IsRunning d) (IsOutdoors d))) (forall ((e BoundSet)) (=> (IsJumping e) (IsOutdoors e))))) (exists ((a BoundSet)) (IsOutdoors a)))))
(check-sat)
(get-model)