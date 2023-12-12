(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsStandingNear (BoundSet BoundSet) Bool)
(declare-fun HasWhiteHair (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (IsStandingNear b a))) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsStandingNear c a) (HasWhiteHair c)))))))
(check-sat)
(get-model)