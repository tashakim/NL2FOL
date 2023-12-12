(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsWaitingFor (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (IsWaitingFor a b))) (exists ((c BoundSet)) (exists ((d BoundSet)) (IsWaitingFor c d))))))
(check-sat)
(get-model)