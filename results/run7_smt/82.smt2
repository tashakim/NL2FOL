(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsWorthTheMoney (BoundSet) Bool)
(declare-fun IsJaneFonda (BoundSet) Bool)
(declare-fun IsInGreatShape (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsWorthTheMoney b) (IsJaneFonda a)))) (forall ((d BoundSet)) (=> (IsInGreatShape d) (IsJaneFonda d)))) (exists ((a BoundSet)) (IsInGreatShape a)))))
(check-sat)
(get-model)