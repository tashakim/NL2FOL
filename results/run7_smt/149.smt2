(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsOnHisKnees (BoundSet) Bool)
(declare-fun IsOnAmusementSlide (BoundSet) Bool)
(declare-fun IsOnSlide (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsOnHisKnees a) (IsOnAmusementSlide b)))) (forall ((e BoundSet)) (forall ((f BoundSet)) (=> (IsOnSlide e) (IsOnAmusementSlide f))))) (exists ((a BoundSet)) (IsOnSlide a)))))
(check-sat)
(get-model)