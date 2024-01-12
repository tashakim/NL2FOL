(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsCrucialTo (BoundSet BoundSet) Bool)
(declare-fun IsWithout (BoundSet BoundSet) Bool)
(declare-fun AreLargestTreesOnEarth (BoundSet) Bool)
(declare-fun IsOnEarth (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (IsCrucialTo a b))) (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (IsWithout a c) (or (AreLargestTreesOnEarth d) (not (IsOnEarth d))))))))))
(check-sat)
(get-model)