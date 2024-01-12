(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsAtLevels (BoundSet BoundSet) Bool)
(declare-fun IsExpectedIn2050 (BoundSet) Bool)
(declare-fun IsPredicted (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsAtLevels a b) (IsExpectedIn2050 c))))) (exists ((d BoundSet)) (not (IsPredicted d))))))
(check-sat)
(get-model)