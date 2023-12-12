(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsBelieveIn (BoundSet) Bool)
(declare-fun IsGod (BoundSet) Bool)
(declare-fun IsForever (BoundSet) Bool)
(declare-fun IsBurnIn (BoundSet) Bool)
(declare-fun IsInHell (BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsBelieveIn a) (not (IsGod b))))) (forall ((e BoundSet)) (forall ((f BoundSet)) (=> (IsGod e) (IsForever f))))) (exists ((d BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsBurnIn a) (and (IsInHell c) (IsForever d)))))))))
(check-sat)
(get-model)