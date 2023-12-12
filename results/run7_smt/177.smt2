(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsPlayingWith (BoundSet BoundSet) Bool)
(declare-fun IsLiftedIntoTheAir (BoundSet) Bool)
(declare-fun IsWearingNoShirt (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (IsPlayingWith b a))) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsLiftedIntoTheAir a) (IsWearingNoShirt c)))))))
(check-sat)
(get-model)