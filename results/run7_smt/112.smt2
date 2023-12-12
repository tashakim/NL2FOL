(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsRiding (BoundSet) Bool)
(declare-fun IsInAlley (BoundSet) Bool)
(declare-fun IsWithDumpster (BoundSet) Bool)
(declare-fun ( (Bool) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsRiding a)) (exists ((d BoundSet)) (exists ((e BoundSet)) (( (and (IsInAlley d) (IsWithDumpster e))))))))
(check-sat)
(get-model)