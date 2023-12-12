(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsWearingBrown (BoundSet) Bool)
(declare-fun IsWearingBlack (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsWearingBrown a) (IsWearingBlack b)))) (exists ((d BoundSet)) (or (IsWearingBrown d) (IsWearingBlack d))))))
(check-sat)
(get-model)