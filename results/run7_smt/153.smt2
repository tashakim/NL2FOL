(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsYoung (BoundSet) Bool)
(declare-fun IsOnBusyStreet (BoundSet) Bool)
(declare-fun IsAtNight (BoundSet) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsYoung a) (and (IsOnBusyStreet b) (IsAtNight c)))))) (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (IsAtNight g) (IsOutside h))))) (exists ((d BoundSet)) (exists ((e BoundSet)) (exists ((f BoundSet)) (and (IsYoung d) (and (IsOutside e) (IsAtNight f)))))))))
(check-sat)
(get-model)