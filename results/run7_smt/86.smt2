(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsProtected (BoundSet) Bool)
(declare-fun IsSpared (BoundSet) Bool)
(declare-fun HasCancer (BoundSet) Bool)
(declare-fun HasOtherIllnesses (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (IsProtected a)) (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((d BoundSet)) (and (IsSpared b) (or (HasCancer c) (HasOtherIllnesses d)))))))))
(check-sat)
(get-model)