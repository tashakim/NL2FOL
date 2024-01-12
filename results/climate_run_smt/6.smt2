(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsOver65 (BoundSet) Bool)
(declare-fun IsGlobalPopulation (BoundSet) Bool)
(declare-fun IsPeople (BoundSet) Bool)
(declare-fun WillHaveIncrease (BoundSet BoundSet) Bool)
(declare-fun IsIn2070 (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsOver65 c) (and (IsGlobalPopulation a) (IsPeople b)))))) (and (forall ((i BoundSet)) (forall ((h BoundSet)) (forall ((g BoundSet)) (=> (IsOver65 g) (WillHaveIncrease h i))))) (forall ((k BoundSet)) (forall ((j BoundSet)) (forall ((l BoundSet)) (=> (WillHaveIncrease j k) (IsOver65 l))))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (exists ((f BoundSet)) (and (WillHaveIncrease d e) (IsIn2070 f))))))))
(check-sat)
(get-model)