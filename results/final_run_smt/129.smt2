(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInRed (BoundSet) Bool)
(declare-fun IsInYellow (BoundSet) Bool)
(declare-fun IsInBlack (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(declare-fun IsWearingUniform (BoundSet) Bool)
(declare-fun IsOnBike (BoundSet) Bool)
(declare-fun IsDoingTricks (BoundSet) Bool)
(assert (not (=> (and (exists ((d BoundSet)) (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((e BoundSet)) (and (exists ((f BoundSet)) (( (and (IsInRed c) (and (IsInYellow d) (IsInBlack e))))) (and (IsWearingUniform f) (IsOnBike b))))))) (forall ((h BoundSet)) (forall ((i BoundSet)) (=> (IsOnBike h) (IsDoingTricks i))))) (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsDoingTricks a) (IsOnBike b)))))))
(check-sat)
(get-model)