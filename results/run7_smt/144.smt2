(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsSittingAlone (BoundSet) Bool)
(declare-fun IsInParkBench (BoundSet) Bool)
(declare-fun IsInSun (BoundSet) Bool)
(declare-fun IsInPark (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsSittingAlone a) (and (IsInParkBench b) (IsInSun c)))))) (exists ((a BoundSet)) (IsInPark a)))))
(check-sat)
(get-model)