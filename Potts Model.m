(* ::Package:: *)

<<"C:/Users/Julian/Documents/GitHub/Machine-Vision/MVTools.m"


isingLabels=2;


DispState[PottsModel[_,labels_,space_]]:=Show[(space/labels)//DispImage,ImageSize->Small]


CreatePottsModel[J_,labels_,{sy_,sx_}]:=PottsModel[J,labels,Table[RandomInteger[labels-1],{j,1,sy},{i,1,sx}]]


LocalEnergyFunction[PottsModel[_,_,space_],j_,i_]:=-Sum[Boole[space[[j,i]]==space[[nj,ni]]],{nj,Max[1,j-1],Min[Length[space],j+1]},{ni,Max[1,i-1],Min[Length[space[[1]]],i+1]}]+1


GibbsSampleOnLattice[PottsModel[J_,labels_,space_?MatrixQ],LocalEnergyFunction_]:=(
   sp=space;
   For[j=1,j<=Length[space],j++,
      For[i=1,i<=Length[space[[1]]],i++,
         tp1=J*LocalEnergyFunction[PottsModel[J,labels,sp],j,i];
         sp2=ReplacePart[sp,{j,i}->RandomInteger[labels-1]];
         tp2=J*LocalEnergyFunction[PottsModel[J,labels,sp2],j,i];
         {p1,p2}={Exp[tp1],Exp[tp2]}/(Exp[tp1]+Exp[tp2]);
         sp=If[p1>Random[],sp2,sp]
   ]];PottsModel[J,labels,sp])


GibbsSampleOnLattice[space_,labels_]:=GibbsSampleOnLattice[space,labels,LocalEnergyFunction]


(*out=NestList[GibbsSampleOnLattice[#,10]&,CreateSpace[10],250];//AbsoluteTiming*)
