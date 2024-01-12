(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(set-option :finite-model-find true)
(declare-fun IsPublished (BoundSet) Bool)
(declare-fun IsAboutOceanAcidification (BoundSet) Bool)
(declare-fun IsACollectionOf (BoundSet) Bool)
(declare-fun IsFlawed (BoundSet) Bool)
(declare-fun IsMethodology (BoundSet) Bool)
(assert (not (=> (exists ((c BoundSet)) (exists ((a BoundSet)) (exists ((b BoundSet)) (and (IsPublished a) (and (IsAboutOceanAcidification b) (IsACollectionOf c)))))) (exists ((e BoundSet)) (exists ((d BoundSet)) (and (IsFlawed e) (and (IsMethodology d) (IsFlawed e))))))))
(check-sat)
(get-model)