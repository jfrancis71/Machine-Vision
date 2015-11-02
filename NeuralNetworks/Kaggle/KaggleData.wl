(* ::Package:: *)

(* ::Input:: *)
(*data=Import["C:\\Users\\Julian\\ImageDataSetsPublic\\FacialKeypoints\\training\\training.csv"];*)


(* ::Input:: *)
(*faceImages=Table[(Partition[ReadList[StringToStream[data[[t,31]]],Number],96]/255.),{t,2,7049}];*)


(* ::Input:: *)
(*faceKeypoints=data[[2;;-1,1;;30]];*)


(* ::Input:: *)
(*faceKeypoints//Length*)


(* ::Input:: *)
(*drawFace[face_,keyP_]:=*)
(*Show[face//Image,Graphics[{Red,Point[{keyP[[1]],96-keyP[[2]]}],Point[{keyP[[3]],96-keyP[[4]]}]}]]*)


(* ::Input:: *)
(*faceKeypoints[[1]]*)


(* ::Input:: *)
(*Table[drawFace[faceImages[[f]],faceKeypoints[[f]]],{f,41,50}]*)


(* ::Input:: *)
(*ImageResize[Image[faceImages[[5]]],{32,32}]*)
