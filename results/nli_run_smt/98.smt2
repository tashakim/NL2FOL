(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsYoungAdult (BoundSet) Bool)
(declare-fun HasX (BoundSet) Bool)
(declare-fun IsAdult (BoundSet) Bool)
(declare-fun IsCryingHysterically (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsYoungAdult a) (HasX b)))) (forall ((e BoundSet)) (forall ((d BoundSet)) (=> (HasX d) (IsAdult e))))) (exists ((c BoundSet)) (and (IsAdult c) (IsCryingHysterically c))))))
(check-sat)
(get-model)