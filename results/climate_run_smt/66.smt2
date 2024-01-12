(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsPredictions (BoundSet) Bool)
(declare-fun IsWorst (BoundSet) Bool)
(declare-fun IsPrepared (BoundSet) Bool)
(declare-fun IsBad (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (IsPredictions b)) (exists ((c BoundSet)) (exists ((d BoundSet)) (and (IsWorst d) (or (IsPrepared c) (IsBad d))))))))
(check-sat)
(get-model)