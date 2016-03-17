(* ::Package:: *)

(* Zero Whitening Transform *)


GenerateZeroWhitening[patches_]:=(
   flPatches=Map[Flatten,patches];
   {u,d,vt}=SingularValueDecomposition[Transpose[flPatches][[All,1;;5000]]];
   forwardTransform=u.Inverse[d[[1;;13*13,1;;13*13]]].Inverse[u];
   backTransform=Inverse[forwardTransform];
   Export[$NNModelDir<>"\\Statistics\\ZeroWhitening.wdx",{forwardTransform,backTransform}];
)


{forwardTransform,backTransform}=Import[$NNModelDir<>"\\Statistics\\ZeroWhitening.wdx"];


encodePatch[patch_]:=Partition[forwardTransform.Flatten[patch],13]


decodePatch[patch_]:=Partition[backTransform.Flatten[patch],13]


encodeImage[image_]:=Flatten[Map[encodePatch,Partition[image,{13,13}],{2}],{{1,3},{2,4}}];


decodeImage[image_]:=Flatten[Map[decodePatch,Partition[image,{13,13}],{2}],{{1,3},{2,4}}];
