(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsInOrange (BoundSet) Bool)
(declare-fun IsRefutingArguments (BoundSet) Bool)
(declare-fun LaunchesCounterargument (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsInOrange a) (IsRefutingArguments c)))) (exists ((b BoundSet)) (exists ((a BoundSet)) (exists ((e BoundSet)) (and (IsInOrange a) (LaunchesCounterargument b e))))))))
(check-sat)
(get-model)