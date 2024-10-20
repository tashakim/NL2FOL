(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInLibrary (BoundSet) Bool)
(declare-fun IsCheeredInFrontOf (BoundSet BoundSet) Bool)
(declare-fun IsNearChildren (BoundSet) Bool)
(declare-fun IsCheering (BoundSet) Bool)
(assert (not (=> (and (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsInLibrary b) (IsCheeredInFrontOf a b)))) (and (forall ((f BoundSet)) (forall ((e BoundSet)) (=> (IsInLibrary e) (IsNearChildren f)))) (and (forall ((i BoundSet)) (forall ((h BoundSet)) (forall ((g BoundSet)) (=> (IsCheeredInFrontOf g h) (IsCheering i))))) (forall ((l BoundSet)) (forall ((j BoundSet)) (forall ((k BoundSet)) (=> (IsCheeredInFrontOf j k) (IsNearChildren l)))))))) (exists ((d BoundSet)) (and (IsCheering d) (IsNearChildren d))))))
(check-sat)
(get-model)