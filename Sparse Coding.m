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
