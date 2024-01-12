(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsMale (BoundSet) Bool)
(declare-fun IsSinging (BoundSet) Bool)
(declare-fun IsInPoorlyLitRoom (BoundSet) Bool)
(declare-fun IsIndoors (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsMale a) (and (IsSinging a) (IsInPoorlyLitRoom c))))) (and (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (IsInPoorlyLitRoom f) (IsIndoors g)))) (forall ((i BoundSet)) (forall ((h BoundSet)) (=> (IsIndoors h) (IsInPoorlyLitRoom i)))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (and (IsSinging d) (IsIndoors e)))))))
(check-sat)
(get-model)