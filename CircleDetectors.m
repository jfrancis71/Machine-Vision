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


circleExtKernel=Table[If[(6<Sqrt[y^2+x^2]<=7),1.0,0.0],{y,-8,+8},{x,-8,+8}];


CircleExt=Position[circleExtKernel,1.];
CircleInt=Map[Function[c,SortBy[Position[circleInsideKernel,1.],EuclideanDistance[#,c]&]//First],CircleExt];


(* ::Text:: *)
(*PDF[StudentTDistribution[0,.02,1],d]  =  15.9155/(1+2500. d^2)*)


ShapeCirclePatch1[patch_]:=54-Total[Log[
   0.3*(15.9/(1.+(2500.*(Extract[patch,CircleExt]-Extract[patch,CircleInt])^2 ))) + 0.7
   ]]


AppearanceCircleConvOpt1[pyr_]:=(
   S1 = (1./(2. .0025))MVCorrelatePyramid[pyr^2,circleInsideKernel];
   S2 = (1./(2. .0025))MVCorrelatePyramid[pyr,circleInsideKernel];
   S3 = (1./(2. .0025)) (Position[circleInsideKernel,1.]//Length);

   Log[1/(Sqrt[2 \[Pi]] .05)^81 .886 * Exp[-S1] Exp[S2^2/S3] ( Erf[(S3-S2)/Sqrt[S3]] + Erf[S2/Sqrt[S3]] )]- 81 Log[3]
)


CirclesRecognition1[image_]:=(
   pyr=BuildPyramid[image,{8,8}];
   app=AppearanceCircleConvOpt1[pyr];
   shape=PyramidFilter[ShapeCirclePatch1, pyr, {8,8}];
   res=shape+Clip[app,{-\[Infinity],20}]
)


CirclesRecognitionOutput1[image_]:=
Show[image//DispImage,OutlineGraphics[BoundingRectangles[CirclesRecognition1[image],80.,{8,8}]]]


On[Assert]
