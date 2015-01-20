(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


(* ::Text:: *)
(*Note Surf and EdgeDir both refer to edge directions, but EdgeDir is signed, whereas Surf is not.*)


circleEdgeKernel=Table[If[(5<=Sqrt[y^2+x^2]<=7),1.0,0.0],{y,-8,+8},{x,-8,+8}];


circleInsideKernel=Table[If[(Sqrt[y^2+x^2]<=5),1.0,0.0],{y,-8,+8},{x,-8,+8}];


HNDist[x_,s_]=FullSimplify[PDF[HalfNormalDistribution[s],x],Assumptions->{x>0}];


ShapeCircleConv[edgeMagPyr_]:=
   MVCorrelatePyramid[
     HNDist[edgeMagPyr,1.6]/(HNDist[edgeMagPyr,1.6]+HNDist[edgeMagPyr,11.6]),circleEdgeKernel]


AppearanceCircleConv[pyr_]:=Log[Sum[Exp[-MVCorrelatePyramid[
   (pyr-a)^2,circleInsideKernel]/(2*0.1)],{a,0,1,0.05}]*0.5];

AppearanceCircleConvOpt[pyr_]:=(
   S1 = MVCorrelatePyramid[pyr^2,circleInsideKernel];
   S2 = MVCorrelatePyramid[pyr,circleInsideKernel];
   S3 = Position[circleInsideKernel,1.]//Length;

   -1+Log[Exp[-S1] Exp[S2^2/S3] ( Erf[(S3-S2)/Sqrt[S3]] + Erf[S2/Sqrt[S3]] )]
)


CirclesRecognitionOutput[img_] := (
   pyr = BuildPyramid[img,{8,8}]; (* 010 *)
   rawShape=ShapeCircleConv[SobelFilter[pyr,EdgeMag]]; (* 130 *)
   shape=1/(1+Exp[-(rawShape-70)*10]);
   rawApp=AppearanceCircleConvOpt[pyr]; (* 120 *)
   appearance=1/(1+Exp[-(rawApp+0.3)*10]);
   Show[img//DispImage,OutlineGraphics[BoundingRectangles[shape*appearance,0.10,{8,8}]] (* 070 *)
]) (* 350 *)


On[Assert]
