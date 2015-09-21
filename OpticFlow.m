(* ::Package:: *)

(* ::Input:: *)
(*MatchDetector[kernel_,frame_]:=*)
(*Module[{c=*)
(*ImageCorrelate[frame//Image,kernel,SquaredEuclideanDistance]//ImageData},*)
(*Exp[-c]/Total[Exp[-c],2]]*)


(* ::Input:: *)
(*InterestDetector[{y_,x_},frame_]:=Max[MatchDetector[frame[[y-5;;y+5,x-5;;x+5]],frame]]*)


(* ::Input:: *)
(*InterestDetector[frame_]:=*)
(*ArrayPad[Table[InterestDetector[{y,x},frame],{y,6,Length[frame]-5},{x,6,Length[frame[[1]]]-5}],5]*)


(* ::Input:: *)
(*FeatureDetector[frame_]:=*)
(*( *)
(*int=InterestDetector[frame];*)
(*m=MorphologicalComponents[int//Reverse,.02];*)
(*c=ComponentMeasurements[{m,int//Image},"Centroid"]*)
(*)*)


(* ::Input:: *)
(*OpticFlow[currentFrame_,previousFrame_]:=( *)
(*conf=ArrayPad[Table[Max[MatchDetector[previousFrame[[y-5;;y+5,x-5;;x+5]],currentFrame]],{y,6,Length[currentFrame]-5},{x,6,Length[currentFrame[[1]]]-5}],5];*)
(*flow=ArrayPad[Table[First[Position[MatchDetector[previousFrame[[y-5;;y+5,x-5;;x+5]],currentFrame],conf[[y,x]]]]-{y,x},{y,6,Length[currentFrame]-5},{x,6,Length[currentFrame[[1]]]-5}],{5,5,0},{0,0}];*)
(*{flow,conf})*)


(* ::Input:: *)
(*DispFeatures[frame_]:=*)
(*Show[frame//DispImage,Graphics[{Red,Map[Circle[#[[2]],5]&,FeatureDetector[frame]]}]]*)


(* ::Input:: *)
(*DispOpticFlow[{flow_,conf_},frame_]:=( *)
(*m=MorphologicalComponents[conf//Reverse,.008];*)
(*c=Map[Round[#[[2]]]&,ComponentMeasurements[{m,conf//Image},"Centroid"]];*)
(*Show[frame//DispImage,Graphics[{Red,Map[Line[{#,#+Reverse[Extract[flow,Reverse[#]]]}]&,c]}]]*)
(*)*)
