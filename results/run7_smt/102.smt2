(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsHispanic (BoundSet) Bool)
(declare-fun IsWearingRedPlaidShirt (BoundSet) Bool)
(declare-fun WorksOnSewing (BoundSet) Bool)
(declare-fun IsWearing (BoundSet) Bool)
(declare-fun IsSewing (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsHispanic a) (and (IsWearingRedPlaidShirt b) (WorksOnSewing c)))))) (and (forall ((g BoundSet)) (forall ((h BoundSet)) (=> (IsWearingRedPlaidShirt g) (IsWearing h)))) (and (forall ((i BoundSet)) (forall ((j BoundSet)) (=> (IsWearing i) (IsWearingRedPlaidShirt j)))) (and (forall ((k BoundSet)) (forall ((l BoundSet)) (=> (WorksOnSewing k) (IsSewing l)))) (forall ((m BoundSet)) (forall ((n BoundSet)) (=> (IsSewing m) (WorksOnSewing n)))))))) (exists ((e BoundSet)) (exists ((f BoundSet)) (and (IsWearing e) (IsSewing f)))))))
(check-sat)
(get-model)