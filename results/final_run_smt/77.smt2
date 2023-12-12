(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsRaisedGoodPoint (BoundSet) Bool)
(declare-fun IsTrusted (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsRaisedGoodPoint a)) (exists ((b BoundSet)) (not (IsTrusted b))))))
(check-sat)
(get-model)