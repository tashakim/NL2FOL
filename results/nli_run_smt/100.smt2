(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsTalkWith (BoundSet BoundSet) Bool)
(declare-fun IsInFrontOfTeam (BoundSet) Bool)
(declare-fun IsInCrowd (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (IsTalkWith a b))) (forall ((f BoundSet)) (forall ((e BoundSet)) (forall ((d BoundSet)) (=> (IsTalkWith d e) (IsInFrontOfTeam f)))))) (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsTalkWith a b) (and (IsInFrontOfTeam c) (IsInCrowd c)))))))))
(check-sat)
(get-model)