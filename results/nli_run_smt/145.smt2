(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsVisiting (BoundSet BoundSet) Bool)
(declare-fun IsInHospital (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsVisiting a c) (IsInHospital b))))) (exists ((a BoundSet)) (exists ((d BoundSet)) (and (IsInHospital a) (IsVisiting a d)))))))
(check-sat)
(get-model)