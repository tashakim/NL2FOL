(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun b () BoundSet)
(declare-fun c () BoundSet)
(declare-fun a () BoundSet)
(declare-fun IsInSpookyCabin (BoundSet) Bool)
(declare-fun ShouldBreakInto (BoundSet BoundSet) Bool)
(assert (not (=> (IsInSpookyCabin b) (and (ShouldBreakInto c a) (forall ((a BoundSet)) (forall ((c BoundSet)) (=> (ShouldBreakInto c a) (IsInSpookyCabin a))))))))
(check-sat)
(get-model)