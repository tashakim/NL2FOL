(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsOlder (BoundSet) Bool)
(declare-fun IsInCrowd (BoundSet) Bool)
(declare-fun IsOnPublicStreet (BoundSet) Bool)
(declare-fun IsStandingOnEdge (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (IsOlder a) (IsInCrowd a))) (exists ((a BoundSet)) (and (IsOnPublicStreet a) (IsStandingOnEdge a))))))
(check-sat)
(get-model)