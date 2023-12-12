(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun AllBoys (BoundSet) Bool)
(declare-fun IsMean (BoundSet) Bool)
(declare-fun IsExperiencedWith (BoundSet BoundSet) Bool)
(assert (not (=> (exists ((b BoundSet)) (exists ((a BoundSet)) (and (AllBoys b) (IsMean a)))) (exists ((d BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsExperiencedWith a c) (IsMean d))))))))
(check-sat)
(get-model)