(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsExperimentedOn (BoundSet) Bool)
(declare-fun IsRespected (BoundSet) Bool)
(declare-fun IsBecomeBattlefield (BoundSet) Bool)
(declare-fun IsIllegal (BoundSet) Bool)
(declare-fun IsEnded (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsExperimentedOn a) (not (IsRespected b))))) (exists ((d BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsBecomeBattlefield c) (and (IsIllegal a) (IsEnded d)))))))))
(check-sat)
(get-model)