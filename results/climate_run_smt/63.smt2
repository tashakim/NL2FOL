(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsAccelerated (BoundSet) Bool)
(declare-fun IsGoodNewsFor (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (not (IsAccelerated c))) (exists ((a BoundSet)) (IsGoodNewsFor a)))))
(check-sat)
(get-model)