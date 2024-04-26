(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsInHugeField (BoundSet) Bool)
(declare-fun IsBrown (BoundSet) Bool)
(declare-fun IsRidingInCar (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (IsInHugeField a) (IsBrown a))) (exists ((c BoundSet)) (IsRidingInCar c)))))
(check-sat)
(get-model)