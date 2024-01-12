(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsBlond (BoundSet) Bool)
(declare-fun IsBehind (BoundSet) Bool)
(declare-fun IsSmiling (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsBlond a) (and (IsBehind b) (not (IsSmiling a b)))))) (exists ((c BoundSet)) (and (IsBehind c) (IsBlond c))))))
(check-sat)
(get-model)