(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsHadExperience (BoundSet) Bool)
(declare-fun IsBoyfriend (BoundSet) Bool)
(declare-fun IsMean (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsHadExperience a) (IsBoyfriend b)))) (exists ((c BoundSet)) (IsMean c)))))
(check-sat)
(get-model)