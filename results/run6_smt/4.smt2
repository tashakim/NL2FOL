(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsTeacherOf (BoundSet) Bool)
(declare-fun IsBestClass (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((b BoundSet)) (and (IsTeacherOf b) (IsBestClass c)))) (exists ((a BoundSet)) (IsBestClass a)))))
(check-sat)
(get-model)