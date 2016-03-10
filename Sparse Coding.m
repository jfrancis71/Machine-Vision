(* ::Package:: *)

(*
   Ref: Iasonas Kokkinos, http://cvn.ecp.fr/personnel/iasonas/course/DL3.pdf, slide 20
   Ref: Hinton, PDP, p. 91, Coarse Coding
*)


(*
   We are assuming uncorrelated sparse codes here.
   If concerned about correlation simply randomly permute the code first
(and apply reverse permutation on decoding
*)
encode[dat_]:=Append[Flatten[Map[
   Switch[#,
      {0,0,0},{0,0},
      {1,0,0},{0,1},
      {0,1,0},{1,0},
      {0,0,1},{1,1},
      _,$Failed]&
   ,Partition[dat,3]]
   ],dat[[1024]]]


decode[sparse_]:=Append[Flatten[Map[
   Switch[#,
      {0,0},{0,0,0},
      {0,1},{1,0,0},
      {1,0},{0,1,0},
      {1,1},{0,0,1}]&,
   Partition[sparse,2]]],sparse[[-1]]]


CheckSparseCode[]:=(
(* Probability of code block failure = 3*1/(1024*1024)
 Probability of failure in any code block .1%
*)
Table[
   sparseDat=Boole[Table[Random[]<1/1024,{1024}]];
   decode[encode[sparseDat]]==sparseDat,{10000}])


(* Emprical Coding error rate 6 per million (on encoding two items) *)


(*
   Ideas from https://research-repository.st-andrews.ac.uk/bitstream/10023/2994/1/FoldiakSparseHBTNN2e02.pdf 
Peter Foldiak: Sparse Coding in the primary cortex (2002) 
http://arxiv.org/ftp/arxiv/papers/1503/1503.07469.pdf
Jeff Hawkins and Subutai Ahmed (2015)
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralLayers.m"


codebook=Table[RandomSample[Join[ConstantArray[1,10],ConstantArray[0,90]]],{1024}];


codebook//Dimensions


{1024,100}


encode[items_List]:=Clip[Total[Map[Function[item,codebook[[item]]],items]],{0,1}]


probItemLog[code_,item_]:=Log[.002]-Log[.998]+(-1000*10)+Extract[code,Position[codebook[[item]],1]].Table[Log[10]+1000,{10}]


probItem1[code_,item_]:=LogisticFn[probItemLog[code,item]]


decode[code_]:=Position[Table[probItem1[code,i],{i,1,1024}],x_/;x>.5][[All,1]]


Count[encode[{45,89}],1]


20


randItems=Table[RandomInteger[1024],{3}]


{234,752,550}


cf=encode[randItems];


cf


{0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0}


decode[cf]


{234,550,752}


test=Flatten[Table[{a,b},{a,1,1024},{b,Delete[Table[r,{r,1,1024}],a]}],1];


Dynamic[{case,sum}]


Dynamic[{$CellContext`case, $CellContext`sum}, ImageSizeCache -> {62., {2., 7.}}]


sum=0;Table[sum += 1-Boole[decode[encode[test[[case]]]]==Sort[test[[case]]]],{case,1,Length[test]}];
