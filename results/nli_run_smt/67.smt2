(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsMany (BoundSet) Bool)
(declare-fun IsAMan (BoundSet) Bool)
(declare-fun IsNaked (BoundSet) Bool)
(declare-fun IsPicking (BoundSet) Bool)
(assert (not (=> (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsMany b) (IsAMan a)))) (exists ((c BoundSet)) (exists ((d BoundSet)) (and (IsNaked c) (IsPicking d)))))))
(check-sat)
(get-model)