(* ::Package:: *)

images=ReadImagesFromDirectory["C:\\Users\\Julian\\Google Drive\\Personal\\Pictures\\Iphone Pictures\\15092014"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


blankPatches=Select[patches,StandardDeviation[Flatten[#]]<.05&];


(*These both come from the top level CircleDetectors file*)


covar = Table[3 Exp[-2.7 Sin[\[Pi] (x1-x2)^1]^2],{x1,0,1-(1/6),1/6},{x2,0,1-(1/6),1/6}];


ShapeFunc[x3_,shape_]:=Table[3 Exp[-2.7 Sin[\[Pi] (x3-x1)]^2],{x1,0,1-(1/6),1/6}].Inverse[covar].shape;


w=ShapeFunc[x,RandomVariate[MultinormalDistribution[ConstantArray[0,6],covar]]];


randomPositive[bg_,fg_]:=( 
   w=ShapeFunc[x,RandomVariate[MultinormalDistribution[ConstantArray[0,6],covar]]];
   Table[If[Norm[{x,y}]<7+w /. x->ArcTan[x,y]/(2 \[Pi]),blankPatches[[fg,1+Round[y+15.5],1+Round[x+15.5]]],patches[[bg,1+Round[y+15.5],1+Round[x+15.5]]]],{y,-15.5,+15.5},{x,-15.5,+15.5}])


randomSamples=Table[If[Random[]>.5,{randomPositive[1+RandomInteger[780-1],1+RandomInteger[60-1]],1},{patches[[1+RandomInteger[780-1]]],0}],{o,1,5000}];


Export["C:\\Users\\Julian\\ImageDataSetsPublic\\Circles\\Circles.wdx",randomSamples]
