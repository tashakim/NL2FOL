(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsJumpedOff (BoundSet) Bool)
(declare-fun HasCityView (BoundSet) Bool)
(declare-fun IsOnCliff (BoundSet) Bool)
(declare-fun IsWalksDown (BoundSet) Bool)
(declare-fun IsOnHill (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsJumpedOff a) (and (HasCityView b) (IsOnCliff c)))))) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (IsWalksDown a) (IsOnHill d)))))))
(check-sat)
(get-model)