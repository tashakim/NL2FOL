(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsRides (BoundSet BoundSet) Bool)
(declare-fun IsOutdoors (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (IsRides a b))) (forall ((e BoundSet)) (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (IsRides e f) (IsOutdoors g)))))) (exists ((d BoundSet)) (IsOutdoors d)))))
(check-sat)
(get-model)