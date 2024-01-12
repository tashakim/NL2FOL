(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsOnIce (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (IsOnIce b)) (exists ((c BoundSet)) (IsOnIce c)))))
(check-sat)
(get-model)