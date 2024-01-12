(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInBlack (BoundSet) Bool)
(declare-fun PlaysHarmonica (BoundSet) Bool)
(declare-fun IsPlayingHarmonica (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsInBlack b) (PlaysHarmonica a)))) (and (forall ((d BoundSet)) (=> (PlaysHarmonica d) (IsPlayingHarmonica d))) (forall ((e BoundSet)) (=> (IsPlayingHarmonica e) (PlaysHarmonica e))))) (exists ((a BoundSet)) (IsPlayingHarmonica a)))))
(check-sat)
(get-model)