(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsMeaningfullyAveraged (BoundSet) Bool)
(declare-fun IsExistent (BoundSet) Bool)
(declare-fun IsGlobalTemperature (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (not (IsMeaningfullyAveraged a))) (exists ((c BoundSet)) (exists ((b BoundSet)) (and (IsExistent b) (not (IsGlobalTemperature c))))))))
(check-sat)
(get-model)