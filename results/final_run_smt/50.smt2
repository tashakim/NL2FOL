(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsAt (BoundSet BoundSet) Bool)
(declare-fun HasAnIPhone (BoundSet) Bool)
(declare-fun ShouldBuy (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsAt b c) (HasAnIPhone a))))) (forall ((e BoundSet)) (forall ((f BoundSet)) (=> (ShouldBuy e f) (HasAnIPhone e))))) (exists ((d BoundSet)) (exists ((a BoundSet)) (ShouldBuy a d))))))
(check-sat)
(get-model)