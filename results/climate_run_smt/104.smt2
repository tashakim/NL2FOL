(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsGettingMoneyOut (BoundSet) Bool)
(declare-fun IsAtATM (BoundSet) Bool)
(declare-fun IsOnTheLeft (BoundSet) Bool)
(declare-fun HasTwoChildren (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsGettingMoneyOut a) (IsAtATM b)))) (exists ((e BoundSet)) (exists ((d BoundSet)) (and (IsOnTheLeft d) (HasTwoChildren e)))))))
(check-sat)
(get-model)