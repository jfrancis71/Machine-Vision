(* ::Package:: *)

(* CATDistributions is a list of categorical distributions, eg n*m matrix where n examples and m categories *)
(* Note there is no single item classification version. That is intentional *)
(* to encourage vectorised operations and discourage loopy styles of coding *)
(* note, labels is the index of the correct label into the categorical distribution *)

CATCrossEntropy[ labels_, CATDistributions_ ] :=
   -Total[ Log[ 2, MapThread[ #1[[#2]]&, {CATDistributions,labels} ] ] ]


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

CATTraining[reset_,CrossEntropyFunction_] := (
  iter = 0;
  Print[Dynamic[{iter,crossEntropy[[-5;;-1]]}]];
  If[reset, w = winitial;crossEntropy={}];
  While[True,
   iter++;
   crossEntropy = Append[crossEntropy,CrossEntropyFunction[w]];
   w = GradientDescent[w];]
  )
      

Test[] := Total[MapThread[
     Boole[ClassifyImage[#1] == #2] &,
     {TestImages, TestLabels}]]/
   Length[TestImages] // N

