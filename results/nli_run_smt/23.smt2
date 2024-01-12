(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsStandingIn (BoundSet BoundSet) Bool)
(declare-fun IsSitting (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (IsStandingIn a b))) (exists ((c BoundSet)) (IsSitting c)))))
(check-sat)
(get-model)