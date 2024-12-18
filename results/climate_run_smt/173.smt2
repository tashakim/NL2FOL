(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInOrangeAndGrayShirt (BoundSet) Bool)
(declare-fun IsFixing (BoundSet) Bool)
(declare-fun IsBeingRepaired (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsInOrangeAndGrayShirt c) (IsFixing a)))) (and (forall ((f BoundSet)) (=> (IsFixing f) (IsBeingRepaired f))) (forall ((g BoundSet)) (=> (IsBeingRepaired g) (IsFixing g))))) (exists ((a BoundSet)) (IsBeingRepaired a)))))
(check-sat)
(get-model)