(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsYawns (BoundSet) Bool)
(declare-fun IsSinging (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsYawns a)) (exists ((a BoundSet)) (exists ((b BoundSet)) (IsSinging a b))))))
(check-sat)
(get-model)