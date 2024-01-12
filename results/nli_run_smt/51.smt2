(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsAdjusting (BoundSet) Bool)
(declare-fun IsLookingAt (BoundSet) Bool)
(declare-fun IsAComputer (BoundSet) Bool)
(declare-fun IsABra (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsAdjusting a) (and (IsLookingAt b) (IsAComputer c)))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (and (IsAdjusting d) (IsABra e)))))))
(check-sat)
(get-model)