(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsHugged (BoundSet BoundSet) Bool)
(declare-fun IsInHospitalRoom (BoundSet) Bool)
(declare-fun IsBorn (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsHugged a b) (IsInHospitalRoom c))))) (and (forall ((h BoundSet)) (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (IsHugged f g) (IsBorn h))))) (forall ((k BoundSet)) (forall ((j BoundSet)) (forall ((i BoundSet)) (=> (IsBorn i) (IsHugged j k))))))) (exists ((d BoundSet)) (IsBorn d)))))
(check-sat)
(get-model)