(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsHaving (BoundSet) Bool)
(declare-fun IsLooking (BoundSet) Bool)
(declare-fun IsSeashells (BoundSet) Bool)
(declare-fun IsHappy (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsHaving a) (and (IsLooking a) (IsSeashells b))))) (and (forall ((c BoundSet)) (=> (IsHaving c) (IsHappy c))) (and (forall ((d BoundSet)) (=> (IsHappy d) (IsHaving d))) (forall ((e BoundSet)) (=> (IsHappy e) (IsLooking e)))))) (exists ((a BoundSet)) (IsHappy a)))))
(check-sat)
(get-model)