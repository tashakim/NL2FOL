(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsCausedBy (BoundSet) Bool)
(declare-fun IsNothingToDoWith (BoundSet BoundSet) Bool)
(declare-fun IsEmittedBy (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsCausedBy a)) (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsNothingToDoWith a b) (not (IsEmittedBy c)))))))))
(check-sat)
(get-model)