(set-logic ALL)
(set-option :produce-models true)
(declare-sort BoundSet 0)
(declare-sort UnboundSet 0)
(declare-fun b () UnboundSet)
(declare-fun c () UnboundSet)
(declare-fun e () UnboundSet)
(declare-fun d () UnboundSet)
(declare-fun a () UnboundSet)
(declare-fun IsGiven (BoundSet UnboundSet) Bool)
(declare-fun IsInLimbo (UnboundSet) Bool)
(declare-fun IsInHeaven (UnboundSet) Bool)
(declare-fun IsLivedLike (BoundSet UnboundSet) Bool)
(declare-fun IsGood (BoundSet) Bool)
(declare-fun IsBeautiful (UnboundSet) Bool)
(declare-fun IsAPlace (BoundSet) Bool)
(declare-fun ~IsLiked (BoundSet BoundSet) Bool)
(declare-fun IsHappy (UnboundSet) Bool)
(declare-fun IsWornBy (UnboundSet) Bool)
(declare-fun IsPartOfFamily (BoundSet) Bool)
(declare-fun IsDone (BoundSet) Bool)
(declare-fun IsSpooky (BoundSet) Bool)
(declare-fun IsMorallyJustifiable (BoundSet) Bool)
(declare-fun IsTeacher (BoundSet) Bool)
(declare-fun IsBestClass (BoundSet) Bool)
(declare-fun IsBest (BoundSet) Bool)
(declare-fun IsClass (BoundSet) Bool)
(declare-fun IsBlonde (UnboundSet) Bool)
(declare-fun IsThere (UnboundSet) Bool)
(declare-fun IsSale (UnboundSet) Bool)
(declare-fun ~IsBelieveIn (UnboundSet) Bool)
(declare-fun IsBurnInHell (UnboundSet) Bool)
(declare-fun IsForever (UnboundSet) Bool)
(declare-fun IsGoingToHaveLunch (BoundSet BoundSet) Bool)
(declare-fun IsBelieve (BoundSet) Bool)
(declare-fun IsDitching (BoundSet) Bool)
(declare-fun IsNew (BoundSet) Bool)
(declare-fun IsCaught (BoundSet UnboundSet) Bool)
(declare-fun IsThrown (UnboundSet UnboundSet) Bool)
(declare-fun IsAt (UnboundSet BoundSet) Bool)
(declare-fun IsPutting (BoundSet BoundSet) Bool)
(declare-fun IsUnder (BoundSet BoundSet) Bool)
(declare-fun IsHispanic (BoundSet) Bool)
(declare-fun WearsRedPlaidShirt (BoundSet) Bool)
(declare-fun IsWearing (BoundSet) Bool)
(declare-fun IsSewing (BoundSet) Bool)
(declare-fun IsAnArticleOfClothing (BoundSet) Bool)
(declare-fun IsWalkedDown (BoundSet UnboundSet) Bool)
(declare-fun IsOutside (BoundSet) Bool)
(declare-fun IsRunTowards (BoundSet BoundSet) Bool)
(declare-fun IsNoShirt (BoundSet) Bool)
(declare-fun IsTampering (BoundSet) Bool)
(declare-fun IsTamperedWith (BoundSet) Bool)
(declare-fun IsMetal (BoundSet) Bool)
(declare-fun IsInGlasses (UnboundSet) Bool)
(declare-fun IsSmiling (UnboundSet) Bool)
(assert (not (=> (=> (IsInGlasses a) (IsSmiling c)) (IsSmiling d))))
(check-sat)
(get-model)