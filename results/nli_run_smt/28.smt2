(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsPlaying (BoundSet BoundSet) Bool)
(declare-fun IsWithSticks (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsPlaying a b) (IsWithSticks c))))) (exists ((a BoundSet)) (exists ((d BoundSet)) (IsPlaying a d))))))
(check-sat)
(get-model)