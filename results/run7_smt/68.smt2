(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsVotingFor (BoundSet) Bool)
(declare-fun President (BoundSet) Bool)
(declare-fun IsNotVotingFor (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsVotingFor a) (President b)))) (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsNotVotingFor c a) (IsVotingFor b))))))))
(check-sat)
(get-model)