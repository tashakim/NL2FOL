(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsCooking (BoundSet) Bool)
(declare-fun IsTalking (BoundSet BoundSet) Bool)
(declare-fun IsInKitchen (BoundSet) Bool)
(assert (not (exists ((b BoundSet)) (IsInKitchen b))))
(check-sat)
(get-model)