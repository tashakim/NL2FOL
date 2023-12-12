(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsKnitted (BoundSet) Bool)
(declare-fun IsFor (BoundSet) Bool)
(declare-fun IsWorn (BoundSet) Bool)
(declare-fun MakesHappy (BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (IsKnitted a) (IsFor b)))) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsWorn c) (MakesHappy a)))))))
(check-sat)
(get-model)