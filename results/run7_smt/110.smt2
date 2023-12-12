(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsFat (BoundSet) Bool)
(declare-fun IsLeaving (BoundSet) Bool)
(declare-fun IsOnLake (BoundSet) Bool)
(declare-fun IsOnDock (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (not (IsFat a))) (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsLeaving a) (and (IsOnLake b) (IsOnDock c)))))))))
(check-sat)
(get-model)