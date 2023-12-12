(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsHaunted (BoundSet) Bool)
(declare-fun IsInOffice (BoundSet) Bool)
(declare-fun IsGhost (BoundSet) Bool)
(declare-fun IsA (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsHaunted b) (and (IsInOffice a) (IsGhost c)))))) (exists ((d BoundSet)) (exists ((e BoundSet)) (not (IsA d e)))))))
(check-sat)
(get-model)