(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsRisingUp (BoundSet) Bool)
(declare-fun IsChanging (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (IsRisingUp b)) (exists ((c BoundSet)) (IsChanging c)))))
(check-sat)
(get-model)