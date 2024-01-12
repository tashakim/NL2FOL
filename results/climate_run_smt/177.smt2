(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsOlderLooking (BoundSet) Bool)
(declare-fun IsInLectureHall (BoundSet) Bool)
(declare-fun IsAtUniversity (BoundSet) Bool)
(declare-fun IsMany (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsOlderLooking b) (and (IsInLectureHall a) (IsAtUniversity c)))))) (exists ((d BoundSet)) (exists ((a BoundSet)) (and (IsMany d) (IsInLectureHall a)))))))
(check-sat)
(get-model)