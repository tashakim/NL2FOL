(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsRed (BoundSet) Bool)
(declare-fun Jumps (BoundSet) Bool)
(declare-fun IsCaughtInMouth (BoundSet) Bool)
(declare-fun IsBall (BoundSet) Bool)
(declare-fun IsCaught (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsRed a) (and (Jumps b) (IsCaughtInMouth c)))))) (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (Jumps f) (IsBall g))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (and (IsCaught d) (IsBall e)))))))
(check-sat)
(get-model)