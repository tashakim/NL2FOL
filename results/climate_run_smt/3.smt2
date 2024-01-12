(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsSettled (BoundSet) Bool)
(declare-fun IsHistoryOfScience (BoundSet) Bool)
(declare-fun HasRepeatedUpheaval (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsSettled c) (IsHistoryOfScience a)))) (exists ((d BoundSet)) (exists ((a BoundSet)) (and (HasRepeatedUpheaval d) (IsHistoryOfScience a)))))))
(check-sat)
(get-model)