(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsDumping (BoundSet) Bool)
(declare-fun IsSoil (BoundSet) Bool)
(declare-fun IsGround (BoundSet) Bool)
(declare-fun IsLazing (BoundSet) Bool)
(declare-fun IsOnBeach (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsDumping a) (and (IsSoil b) (IsGround c)))))) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (IsLazing a) (IsOnBeach d)))))))
(check-sat)
(get-model)