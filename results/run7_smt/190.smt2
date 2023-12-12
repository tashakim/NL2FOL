(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsBrown (BoundSet) Bool)
(declare-fun HasBlueMuzzle (BoundSet) Bool)
(declare-fun IsInField (BoundSet) Bool)
(declare-fun IsWearingMuzzle (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsBrown a) (and (HasBlueMuzzle b) (IsInField c)))))) (forall ((g BoundSet)) (forall ((f BoundSet)) (=> (IsWearingMuzzle f) (HasBlueMuzzle g))))) (exists ((d BoundSet)) (IsWearingMuzzle d)))))
(check-sat)
(get-model)