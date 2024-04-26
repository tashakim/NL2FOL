(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun AreStandingIn (BoundSet) Bool)
(declare-fun IsInBoat (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (AreStandingIn b) (IsInBoat a)))) (exists ((c BoundSet)) (IsInBoat c)))))
(check-sat)
(get-model)