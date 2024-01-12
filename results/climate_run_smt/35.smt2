(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsHonoredByNASA (BoundSet) Bool)
(declare-fun HasAchievement (BoundSet) Bool)
(declare-fun IsInHouseScienceCommittee (BoundSet) Bool)
(declare-fun IsInMarch2017 (BoundSet) Bool)
(declare-fun IsTold (BoundSet) Bool)
(declare-fun AreFailed (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((b BoundSet)) (and (IsHonoredByNASA b) (HasAchievement c)))) (and (forall ((h BoundSet)) (forall ((i BoundSet)) (=> (IsInHouseScienceCommittee h) (HasAchievement i)))) (forall ((k BoundSet)) (forall ((j BoundSet)) (=> (IsInMarch2017 j) (HasAchievement k)))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (exists ((f BoundSet)) (exists ((g BoundSet)) (and (IsTold d) (and (and (IsInHouseScienceCommittee e) (IsInMarch2017 f)) (AreFailed g))))))))))
(check-sat)
(get-model)