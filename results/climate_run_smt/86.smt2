(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsAlarmist (BoundSet) Bool)
(declare-fun ArePoliticians (BoundSet) Bool)
(declare-fun AreEnvironmentalists (BoundSet) Bool)
(declare-fun AreMedia (BoundSet) Bool)
(declare-fun IsMotivatedByIdeology (BoundSet) Bool)
(declare-fun IsMotivatedByMoney (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(declare-fun IsGenuinelyConcerned (BoundSet) Bool)
(assert (not (=> (and (exists ((c BoundSet)) (exists ((d BoundSet)) (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsAlarmist d) (and (ArePoliticians a) (and (AreEnvironmentalists b) (AreMedia c)))))))) (and (forall ((f BoundSet)) (forall ((g BoundSet)) (=> (IsAlarmist f) (IsMotivatedByIdeology g)))) (and (forall ((h BoundSet)) (=> (AreEnvironmentalists h) (IsMotivatedByIdeology h))) (forall ((i BoundSet)) (=> (IsMotivatedByIdeology i) (AreEnvironmentalists i)))))) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (exists ((b BoundSet)) (( (and (IsMotivatedByMoney a) (IsMotivatedByIdeology b)))) (not (IsGenuinelyConcerned c))))))))
(check-sat)
(get-model)