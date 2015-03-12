(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


circleInsideKernel=Table[If[(Sqrt[y^2+x^2]<=5),1.0,0.0],{y,-8,+8},{x,-8,+8}];


AngleDiff[a_,b_]:=Min[Abs[a-b],2 \[Pi] - Abs[a-b]]


circleBoundaryKernel=Table[If[(5<Sqrt[y^2+x^2]<=6),1.0,0.0],{y,-8,+8},{x,-8,+8}];


circleEdgeVKernel=Table[If[x==0&&y==0,0,If[
   AngleDiff[\[Pi]/2,ArcTan[x,y]]<((\[Pi]/2)/3)||AngleDiff[-\[Pi]/2,ArcTan[x,y]]<((\[Pi]/2)/3)
      ,1.,0.]],{y,-8,+8},{x,-8,+8}]*circleBoundaryKernel;


circleEdgeHKernel=Table[If[x==0&&y==0,0,If[
   AngleDiff[\[Pi],ArcTan[x,y]]<((\[Pi]/2)/3)||AngleDiff[0,ArcTan[x,y]]<((\[Pi]/2)/3)
      ,1.,0.]],{y,-8,+8},{x,-8,+8}]*circleBoundaryKernel;


circleEdgeDiag1Kernel=Table[If[x==0&&y==0,0,If[
   AngleDiff[\[Pi]/4,ArcTan[x,y]]<((\[Pi]/2)/3)-.2||AngleDiff[-3 \[Pi]/4,ArcTan[x,y]]<((\[Pi]/2)/3)-.2
      ,1.,0.]],{y,-8,+8},{x,-8,+8}]*circleBoundaryKernel;


circleEdgeDiag2Kernel=Table[If[x==0&&y==0,0,If[
   AngleDiff[3 \[Pi]/4,ArcTan[x,y]]<((\[Pi]/2)/3)-.2||AngleDiff[- \[Pi]/4,ArcTan[x,y]]<((\[Pi]/2)/3)-.2
      ,1.,0.]],{y,-8,+8},{x,-8,+8}]*circleBoundaryKernel;


ShapeCirclePatch[patch_]:=-(
Log[
   0.3*(15.9/(1.+(2500.*patch[[1]] ))) + 0.7
      ]*circleEdgeHKernel+
Log[
   0.3*(15.9/(1.+(2500.*patch[[2]] ))) + 0.7
      ]*circleEdgeVKernel+
Log[
   0.3*(15.9/(1.+(2500.*patch[[3]] ))) + 0.7
      ]*circleEdgeDiag1Kernel+
Log[
   0.3*(15.9/(1.+(2500.*patch[[4]] ))) + 0.7
      ]*circleEdgeDiag2Kernel
)


(* ::Text:: *)
(*PDF[StudentTDistribution[0,.02,1],d]  =  15.9155/(1+2500. d^2)*)


ShapeCircleConv[pyr_]:=(
   horizDiff=MVCorrelatePyramid[pyr,{{1,0,-1}}]^2;
   vertDiff=MVCorrelatePyramid[pyr,Transpose[{{1,0,-1}}]]^2;
   diag1Diff=MVCorrelatePyramid[pyr,{{0,0,1},{0,0,0},{-1,0,0}}//Reverse]^2;
   diag2Diff=MVCorrelatePyramid[pyr,{{0,0,1},{0,0,0},{-1,0,0}}]^2;
   -(
    MVCorrelatePyramid[Log[0.3*15.9/(1+horizDiff*2500)+0.7],circleEdgeHKernel]+
    MVCorrelatePyramid[Log[0.3*15.9/(1+vertDiff*2500)+0.7],circleEdgeVKernel]+
    MVCorrelatePyramid[Log[0.3*15.9/(1+diag1Diff*2500)+0.7],circleEdgeDiag1Kernel]+
    MVCorrelatePyramid[Log[0.3*15.9/(1+diag2Diff*2500)+0.7],circleEdgeDiag2Kernel]
)
)


AppearanceCircleConvOpt[pyr_]:=(
   S1 = (1./(2. .0025)) * MVCorrelatePyramid[pyr^2,circleInsideKernel];
   S2 = (1./(2. .0025)) * MVCorrelatePyramid[pyr,circleInsideKernel];
   S3 = (1./(2. .0025)) * (Position[circleInsideKernel,1.]//Length)//N;

   (*Log[1./(Sqrt[2 3.14] .05)^81 .886 * Exp[-S1] Exp[S2^2/S3] ( Erf[(S3-S2)/Sqrt[S3]] + Erf[S2/Sqrt[S3]] )]- 81 Log[3]*)
   Log[1./(Sqrt[2 3.14] .05)^81] - 81 Log[3] + Log[.886] + -S1 + S2^2/S3 + 0.7
)


CirclesRecognition[image_]:=(
   pyr=BuildPyramid[image,{8,8}];
   app=AppearanceCircleConvOpt[pyr];
   (*shape=PyramidFilter[ShapeCirclePatch1, pyr, {8,8}];*)
   shape=ShapeCircleConv[pyr];
   res=shape+Clip[app,{-\[Infinity],20}]
)


CirclesRecognitionOutput[image_,threshold_:26]:=
   Show[image//DispImage,OutlineGraphics[BoundingRectangles[CirclesRecognition[image],threshold,{8,8}]]]


covar = Table[3 Exp[-2.7 Sin[\[Pi] (x1-x2)^1]^2],{x1,0,1-(1/6),1/6},{x2,0,1-(1/6),1/6}];


ShapeFunc[x3_,shape_]:=Table[3 Exp[-2.7 Sin[\[Pi] (x3-x1)]^2],{x1,0,1-(1/6),1/6}].Inverse[covar].shape;


baseShape=ConstantArray[0,6]


{0,0,0,0,0,0}


ShapePrior[shap_]:=Evaluate[Log[PDF[MultinormalDistribution[baseShape,covar],shap]]]


radius[x_,y_,shape_]:=If[x==0&&y==0,100,5+ShapeFunc[ArcTan[x,y]/(2*3.141),shape]]


App[patch_,shape_]:=
(kern=Table[If[(x1^2+x2^2)<=radius[x1,x2,shape]^2,1,0],{x2,-8,+8},{x1,-8,+8}];
(*
S1 = (1./(2. .0025))Sum[If[(x1^2+x2^2)<=radius[x1,x2,shape]^2,patch[[x2+9,x1+9]]^2,0],{x2,-8,+8},{x1,-8,+8}];
   S2 = (1./(2. .0025))Sum[If[(x1^2+x2^2)<=radius[x1,x2,shape]^2,patch[[x2+9,x1+9]],0],{x2,-8,+8},{x1,-8,+8}];
   S3 = (1./(2. .0025)) Sum[If[(x1^2+x2^2)<=radius[x1,x2,shape]^2,1.,0.],{x2,-8,+8},{x1,-8,+8}];
countP = Sum[If[(x1^2+x2^2)<=radius[x1,x2,shape]^2,1.,0.],{x2,-8,+8},{x1,-8,+8}];*)
countP = Total[kern,2];
S1 = (1./(2. .0025)) Total[patch^2*kern,2];
S2 = (1./(2. .0025)) Total[patch*kern,2];
S3 = (1./(2. .0025)) countP;

   Log[1/(Sqrt[2 \[Pi]] .05)^countP .886 * Exp[-S1] Exp[S2^2/S3] ( Erf[(S3-S2)/Sqrt[S3]] + Erf[S2/Sqrt[S3]] )]- countP Log[3]
)


MVShapeCircleFilt[patch_,shape_]:=-Sum[
If[(radius[x1,x2,shape]+1)^2<(x1^2+x2^2)<(radius[x1,x2,shape]+2)^2,Log[
0.3*(15.9/(1.+(2500.*(patch[[x2+9,x1+9]]-Extract[patch,{9,9}+{x2,x1}-Round[2*Normalize[{x2,x1}]]])^2 ))) + 0.7],0],
{x2,-8,+8},{x1,-8,+8}]


DeformableCircleRecognition[patch_,shape_]:=(
app=App[patch,shape];
shapeOut=MVShapeCircleFilt[patch,shape];
ShapePrior[shape]+Clip[app,{-\[Infinity],20}]+shapeOut)


DeformableCircleRecognition[patch_]:=Table[
shp=baseShape+{s1,s2,s3,s4,s5,s6};
DeformableCircleRecognition[patch,shp],{s1,0,1},{s2,0,+1},{s3,0,1},{s4,0,1},{s5,0,1},{s6,0,1}]


GradientShape[patch_,shape_]:=
{
DeformableCircleRecognition[patch,shape+{1,0,0,0,0,0}],
DeformableCircleRecognition[patch,shape+{0,1,0,0,0,0}],
DeformableCircleRecognition[patch,shape+{0,0,1,0,0,0}],
DeformableCircleRecognition[patch,shape+{0,0,0,1,0,0}],
DeformableCircleRecognition[patch,shape+{0,0,0,0,1,0}],
DeformableCircleRecognition[patch,shape+{0,0,0,0,0,1}]
}-DeformableCircleRecognition[patch,shape]


GradientDescent[patch_,shape_,path_]:=(
g=GradientShape[patch,shape];
If[Length[path]<10,GradientDescent[patch,shape+ReplacePart[ConstantArray[0,6],Last[Ordering[g]]->1],Append[path,{DeformableCircleRecognition[patch,shape],shape}]],path]
)


CircleShapePlot[shape_,ptch_]:=Show[{ptch//DispImage,ParametricPlot[{9,9}+{(5+ShapeFunc[x3,shape])*Cos[x3 2 \[Pi]],(5+ShapeFunc[x3,shape])*Sin[x3 2 \[Pi]]},{x3,0,1},PlotStyle->{Thick,Red}]}]


DeformableCircleRecognitionMax[patch_]:=GradientDescent[patch,ConstantArray[0,6],{}][[All,1]]//Max


CirclesRecognition1[image_]:=(
res=CirclesRecognition[image];
res1 = res;
If[Max[res1]<26,
ptchLoc=Position[res1,Max[res1]]//First;
If[InBoundsQ[pyr,ptchLoc[[1]],ptchLoc[[2]],ptchLoc[[3]]],
   res1[[ptchLoc[[1]],ptchLoc[[2]],ptchLoc[[3]]]] = DeformableCircleRecognitionMax[Patch[pyr,ptchLoc[[1]],ptchLoc[[2]],ptchLoc[[3]]]]+7.8,0],0];
res1
)


CirclesRecognitionOutput1[image_]:=
   Show[image//DispImage,OutlineGraphics[BoundingRectangles[CirclesRecognition1[image],26.,{8,8}]]]


NeighbourProb[x1_,x2_]=0.3*(15.9/(1.+(2500.*(x1-x2)^2)))+0.7;


JArcTan[x_,y_]=If[ArcTan[x,y]<0,2 \[Pi] + ArcTan[x,y],ArcTan[x,y]];


ToPolar[{x_,y_}]={Sqrt[x^2+y^2],ArcTan[x,y]};


ToCartesian[{r_,\[Theta]_}]={Cos[\[Theta]],Sin[\[Theta]]}*r;


JExtract[patch_,coords_]:=If[InBoundsQ[patch,coords[[2]],coords[[1]]],patch[[coords[[2]],coords[[1]]]],-100]


LRF[patch_,fieldNo_,shape_,b_]:=(
kernel=Table[If[y==0&&x==0,0,If[(fieldNo-1)*2*\[Pi]/6<=JArcTan[x,y]<(fieldNo)*2*\[Pi]/6,If[Norm[{x,y}]<shape,1,0],0]],{y,-8,+8},{x,-8,+8}];
appTrue=Table[If[kernel[[y+9,x+9]]==1,Log[1/(Sqrt[2 \[Pi]] 0.1)] - ((patch[[y+9,x+9]]-b)^2/(2 0.1^2)),0],{y,-8,+8},{x,-8,+8}];
appFalse=Total[kernel,2];
edgeKernel=Table[If[y==0&&x==0,0,If[(fieldNo-1)*2*\[Pi]/6<=JArcTan[x,y]<(fieldNo)*2*\[Pi]/6,If[shape<Norm[{x,y}]<shape+2,1,0],0]],{y,-8,+8},{x,-8,+8}];
shapeTab=Log[Table[If[edgeKernel[[y+9,x+9]]==1,NeighbourProb[
JExtract[patch,Round[{9,9}+ToCartesian[ToPolar[{x,y}]-{1,0}]]],JExtract[patch,Round[{9,9}+ToCartesian[ToPolar[{x,y}]+{1,0}]]]],1],{y,-8,+8},{x,-8,+8}]];
ans=Total[appTrue,2]-appFalse-Total[shapeTab,2]
)


IRFGenerator[patch_,fieldNo_,previousS_,b_]:=Sum[
If[fieldNo==1,PDF[NormalDistribution[5,1],currentS],PDF[NormalDistribution[previousS,1],currentS]]*
Exp[LRF[patch,fieldNo,currentS,b]]*
If[fieldNo<6,IRF[fieldNo+1][[Round[currentS]+1]],1]
,{currentS,0,6}]


Eval[patch_,b_]:=(Clear[IRF];For[f=6,f>0,f--,IRF[f]=Table[IRFGenerator[patch,f,previousS,b],{previousS,0,6}]];)


bEval[patch_]:=(Eval[patch,Mean[patch[[8;;10,8;;10]]//Flatten]];IRF[1][[1]])


On[Assert]
