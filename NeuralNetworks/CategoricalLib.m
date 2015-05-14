(* ::Package:: *)

(* Function f takes inputs and returns list of categorical distributions *)
(* Note there is no single item classification version. That is intentional *)
(* to encourage vectorised operations and discourage loopy styles of coding *)
(* note, labels is the index of the correct label into the categorical distribution *)

CATCrossEntropy[ labels_, inputs_, f_ ] := (
   CATDistributions = f[ inputs ];
   -Total[ Log[ 2, MapThread[ #1[[#2]]&, {CATDistributions,labels} ] ] ]
)


(* CATClassify returns a list of the indices of the classification *)
CATClassify[ inputs_, f_ ] := (
   CATDistributions = f[ inputs ];
   Map[ First[ Ordering[ #,-1 ] ]&, CATDistributions ]
)

(* CATTest return percentage classified correctly, indices is the index of the correct label in the category set *)
CATTest[ classificationIndices_, inputs_, f_ ] := (
   classifications = CATClassify[ inputs, f ];
   Total[ MapThread[ Boole[ #1 == #2 ]&, { classifications, classificationIndices } ] ]/Length[ classificationIndices ] //N
)

CATTraining[reset_] := (
  iter = 0; If[reset, w = winitial;energy={}];
  While[True,
   iter++;
   energy = Append[energy,Energy[w]];
   w = GradientDescent[w];]
  )
      

Test[] := Total[MapThread[
     Boole[ClassifyImage[#1] == #2] &,
     {TestImages, TestLabels}]]/
   Length[TestImages] // N

