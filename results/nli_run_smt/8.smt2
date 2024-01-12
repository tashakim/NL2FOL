(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsInBlueShirt (BoundSet) Bool)
(declare-fun IsWorkingOnExerciseBicycleControlPanel (BoundSet) Bool)
(declare-fun IsBald (BoundSet) Bool)
(declare-fun IsTouching (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (and (IsInBlueShirt a) (IsWorkingOnExerciseBicycleControlPanel a))) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (not (IsBald a)) (not (IsTouching a b))))))))
(check-sat)
(get-model)