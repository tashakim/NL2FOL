(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsYoung (BoundSet) Bool)
(declare-fun IsKicked (BoundSet BoundSet) Bool)
(declare-fun IsInBackground (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsYoung a) (IsKicked a b)))) (exists ((c BoundSet)) (IsInBackground c)))))
(check-sat)
(get-model)