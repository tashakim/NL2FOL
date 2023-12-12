(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsAtSchool (BoundSet) Bool)
(declare-fun IsOnFamilyTrip (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (not (IsAtSchool a))) (exists ((c BoundSet)) (IsOnFamilyTrip c)))))
(check-sat)
(get-model)