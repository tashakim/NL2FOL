(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsLarge (BoundSet) Bool)
(declare-fun IsBusy (BoundSet) Bool)
(declare-fun IsAtNight (BoundSet) Bool)
(declare-fun IsInCity (BoundSet) Bool)
(declare-fun IsWalk (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsLarge a) (and (IsBusy b) (IsAtNight c)))))) (and (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (IsBusy f) (IsInCity g)))) (forall ((i BoundSet)) (forall ((h BoundSet)) (=> (IsInCity h) (IsBusy i)))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (and (IsWalk d) (IsInCity e)))))))
(check-sat)
(get-model)