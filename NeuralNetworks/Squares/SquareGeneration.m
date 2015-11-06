(* ::Package:: *)

images=ReadImagesFromDirectory["C:\\Users\\Julian\\Google Drive\\Personal\\Pictures\\Iphone Pictures\\15092014"];


patches=Flatten[Map[Partition[#,{32,32}]&,images],2];


blankPatches=Select[patches,StandardDeviation[Flatten[#]]<.05&];


randomPositive[bg_,fg_]:=( 
ty=6+(Random[]-.5)*2;
by=-6+(Random[]-.5)*2;
lx=-6+(Random[]-.5)*2;
ry=6+(Random[]-.5)*2;

Table[If[y<ty&&y>by&&x>lx&&x<ry,blankPatches[[fg,1+Round[y+15.5],1+Round[x+15.5]]],patches[[bg,1+Round[y+15.5],1+Round[x+15.5]]]],{y,-15.5,+15.5},{x,-15.5,+15.5}])


randomSamples=Table[If[Random[]>.5,{randomPositive[1+RandomInteger[780-1],1+RandomInteger[60-1]],1},{patches[[1+RandomInteger[780-1]]],0}],{o,1,5000}];


Export["C:\\Users\\Julian\\ImageDataSetsPublic\\Squares\\Squares.wdx",randomSamples]
