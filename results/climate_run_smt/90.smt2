(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsInGlobalWarming (BoundSet) Bool)
(declare-fun IsAnImpactOfClimateChange (BoundSet) Bool)
(declare-fun IsAConfluenceOfRareEvents (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((b BoundSet)) (and (IsInGlobalWarming b) (IsAnImpactOfClimateChange c)))) (and (forall ((h BoundSet)) (forall ((g BoundSet)) (=> (IsAnImpactOfClimateChange g) (IsInGlobalWarming h)))) (forall ((j BoundSet)) (forall ((i BoundSet)) (=> (IsInGlobalWarming i) (IsAnImpactOfClimateChange j)))))) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsInGlobalWarming b) (IsAConfluenceOfRareEvents a)))))))
(check-sat)
(get-model)