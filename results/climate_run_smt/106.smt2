(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsWearingOrangeCoat (BoundSet) Bool)
(declare-fun IsWorkingOnStreet (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (or (exists ((b BoundSet)) (( (and (IsWearingOrangeCoat a) (IsWorkingOnStreet b)))) (IsWorkingOnStreet c)))) (forall ((d BoundSet)) (=> (IsWearingOrangeCoat d) (IsOutside d)))) (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsOutside a) (and (IsOutside b) (IsOutside c)))))))))
(check-sat)
(get-model)