(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsWalkingOutside (BoundSet) Bool)
(declare-fun IsAllDressedInWhite (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsWalkingOutside a)) (exists ((a BoundSet)) (IsAllDressedInWhite a)))))
(check-sat)
(get-model)