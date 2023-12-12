(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsExercise (BoundSet) Bool)
(declare-fun LosesWeight (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsExercise a) (LosesWeight b)))) (exists ((b BoundSet)) (not (IsExercise b))))))
(check-sat)
(get-model)