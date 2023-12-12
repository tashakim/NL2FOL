(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun IsSittingIn (BoundSet BoundSet) Bool)
(declare-fun IsInBarberShop (BoundSet BoundSet) Bool)
(declare-fun IsReading (BoundSet BoundSet) Bool)
(assert (not (=> (and (exists ((b BoundSet)) (exists ((a BoundSet)) (IsSittingIn a b))) (and (forall ((d BoundSet)) (forall ((e BoundSet)) (=> (IsSittingIn d e) (IsInBarberShop d e)))) (forall ((g BoundSet)) (forall ((f BoundSet)) (=> (IsInBarberShop f g) (IsSittingIn f g)))))) (exists ((b BoundSet)) (exists ((c BoundSet)) (exists ((a BoundSet)) (and (IsReading a c) (IsInBarberShop a b))))))))
(check-sat)
(get-model)