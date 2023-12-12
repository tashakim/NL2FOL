(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsGettingReady (BoundSet) Bool)
(declare-fun IsABaseballPlayer (BoundSet) Bool)
(declare-fun IsCatchingFlyBall (BoundSet) Bool)
(declare-fun IsNearOutfieldFence (BoundSet) Bool)
(declare-fun IsPlayingBaseball (BoundSet) Bool)
(declare-fun IsOutdoors (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (and (IsGettingReady a) (and (IsABaseballPlayer a) (and (IsCatchingFlyBall a) (IsNearOutfieldFence a))))) (and (forall ((e BoundSet)) (=> (IsABaseballPlayer e) (IsPlayingBaseball e))) (and (forall ((f BoundSet)) (=> (IsPlayingBaseball f) (IsABaseballPlayer f))) (forall ((g BoundSet)) (=> (IsNearOutfieldFence g) (IsOutdoors g)))))) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsPlayingBaseball a) (IsOutdoors c)))))))
(check-sat)
(get-model)