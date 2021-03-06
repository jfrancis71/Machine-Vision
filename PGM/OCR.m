(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


article=StandardiseImage["C:\Users\Julian\Google Drive\Personal\Mobile Pictures\Vision\Article2.jpg",2048];


letterTemplates={
   {"a",article[[1019-6;;1019+5,336-5;;336+5]]},
   {"b",article[[945-6;;945+9,244-5;;244+5]]},
   {"c",article[[969-6;;969+7,263-5;;263+5]]},
   {"d",article[[1019-6;;1019+10,368-5;;368+5]]},
   {"e",article[[1019-6;;1019+5,297-5;;297+5]]},
   {"f",article[[850-6;;850+10,261-4;;261+4]]},
   {"g",article[[1016-7;;1016+10,346-4;;346+5]]},
   {"h",article[[993-7;;993+10,151-4;;151+5]]},
   {"i",article[[1018-7;;1018+10,224-3;;224+3]]},
   {"k",article[[240-5;;240+12,153-3;;153+8]]},
   {"l",article[[850+125-5;;850+125+7,328-4;;328+4]]},
   {"m",article[[655-7;;655+6,260-10;;260+9]]},
   {"n",article[[945-9;;945+5,146-5;;146+6]]},
   {"o",article[[1019-6;;1019+5,245-5;;245+5]]},
   {"p",article[[850+26-15;;850+26+5,238-5;;238+5]]},
   {"q",article[[850+19-15;;850+19+6,55-5;;55+7]]},
   {"r",article[[1019-6;;1019+5,328-5;;326+5]]},
   {"s",article[[850-12;;850+4,120-4;;120+5]]},
   {"t",article[[850-8;;850+7,250-4;;250+4]]},
   {"u",article[[850-13;;850+5,57-5;;57+6]]},
   {"v",article[[1019-6;;1019+5,307-5;;307+5]]},
   {"w",article[[1019-9;;1019+4,174-8;;174+5]]},
   {"y",article[[850+48-15;;850+48+5,193-5;;193+5]]}
};

LettersLength=Length[letterTemplates];


sourceText=Take[Characters[ExampleData[{"Text","OriginOfSpecies"}]],20000];


(*Note I' m using the Laplacian adjustment
Also not quite applicable to letter in a word sequence as this method straddles word bounaries *)

aletterFrequencies=Table[(1+Count[sourceText,letterTemplates[[l,1]]])/(21+1+Sum[Count[sourceText,letterTemplates[[l,1]]],{l,1,LettersLength}]),{l,1,LettersLength}]//N;


freq=Table[Count[Map[#[[2]]&,Select[Partition[sourceText,2],#[[1]]==l1&]],l2],{l1,letterTemplates[[All,1]]},{l2,letterTemplates[[All,1]]}];


aletterConditionalProbs=Table[(1+freq[[l1,l2]])/(LettersLength+1+Total[freq[[l1]]]),{l1,1,LettersLength},{l2,1,LettersLength}]//N;


probF[z_]=FullSimplify[PDF[HalfNormalDistribution[15],z]/PDF[NormalDistribution[0.5,0.15],z],Assumptions->z>0];


letterWidths=Map[Dimensions[#[[2]]][[2]]&,letterTemplates];


letterFrequencies=Table[1,{LettersLength}];


letterConditionalProbs=Table[1,{LettersLength},{LettersLength}];


posLetterConditionalProbs = Table[aletterConditionalProbs[[l2,l1]]*
   Table[If[Abs[((letterWidths[[l1]]+letterWidths[[l2]])/2-d)]<3,1,0],{d,5,20}]
      ,{l2,1,Length[letterTemplates]},{l1,1,Length[letterTemplates]}];


letterKernels = Table[Table[If[Abs[((letterWidths[[l1]]+letterWidths[[l2]])/2-x)]<3,1,0],{x,-20,+20}]
      ,{l2,1,Length[letterTemplates]},{l1,1,Length[letterTemplates]}];
(* Note this is the crude equivelant of previous algo *)
(*letterKernels=Table[Table[If[x>=5,1,0],{x,-20,+20}]
      ,{l2,1,Length[letterTemplates]},{l1,1,Length[letterTemplates]}];*)


(* ::Text:: *)
(*\[Tau] Represents the probability distribution over that state excluding whether we are dealing with a conditional or a prior distribution*)


DispText[y_,letters_,image_]:=Show[
   image//DispImage,
   Graphics[
   {Red,Map[Text[StyleForm[letterTemplates[[#[[1]],1]],FontSize->34],{#[[2]],y}]&,letters]}]]


CondBacktrack1[l2b_,yb_,x2b_]:=
   {{l1b,x1b}=\[Psi]ArgMax[\[Tau]1,l2b,yb,x2b]};


CondBacktrack2[l3b_,yb_,x3b_]:=
(
   {l2b,x2b}=\[Psi]ArgMax[\[Tau]2,l3b,yb,x3b];
   Prepend[CondBacktrack1[l2b,yb,x2b],{l2b,x2b}]
);


CondBacktrack3[l4b_,yb_,x4b_]:=
(
   {l3b,x3b}=\[Psi]ArgMax[\[Tau]2,l4b,yb,x4b];
   Prepend[CondBacktrack2[l3b,yb,x3b],{l3b,x3b}]
);


CondBacktrack4[l5b_,yb_,x5b_]:=
(
   {l4b,x4b}=\[Psi]ArgMax[\[Tau]2,l5b,yb,x5b];
   Prepend[CondBacktrack3[l4b,yb,x4b],{l4b,x4b}]
);


Backtrack1[]:=(
   {yb,l1b,x1b}=Position[\[Phi]1,Max[\[Phi]1]]//First;
   {{l1b,x1b}}
)


Backtrack2[]:=(
   {yb,l2b,x2b}=Position[\[Phi]2,Max[\[Phi]2]]//First;

   Prepend[CondBacktrack1[l2b,yb,x2b],{l2b,x2b}]
)


Backtrack3[]:=(
   {yb,l3b,x3b}=Position[\[Phi]3,Max[\[Phi]3]]//First;

   Prepend[CondBacktrack2[l3b,yb,x3b],{l3b,x3b}]
)


Backtrack4[]:=(
   {yb,l4b,x4b}=Position[\[Phi]4,Max[\[Phi]4]]//First;

   Prepend[CondBacktrack3[l4b,yb,x4b],{l4b,x4b}]
)


Backtrack5[]:=(
   {yb,l5b,x5b}=Position[\[Phi]5,Max[\[Phi]5]]//First;

   Prepend[CondBacktrack4[l5b,yb,x5b],{l5b,x5b}]
)


(*
Returns the max probability configuration conditioned on a letter at a particular position,
ie is a map
*)
\[Psi]Max[\[Tau]_]:=Map[Max,Transpose[Table[(Dilation[\[Tau][[All,l1,All]]//Image,{letterKernels[[l2,l1]]//Reverse}]//ImageData)*letterConditionalProbs[[l2,l1]],
   {l2,1,LettersLength},
   {l1,1,LettersLength}
   ]
   ,{1,4,2,3}],{3}];

(*
Returns the actual next letter configuration that was actually selected to maximise joint probability conditioned on the previous letter *)
\[Psi]ArgMax[\[Tau]_,cl_,cy_,cx_]:=
   (tab=Table[letterKernels[[cl,l,x+21]]*\[Tau][[cy,l,cx+x]]*letterConditionalProbs[[cl,l]],{l,1,LettersLength},{x,-20,+20}];
   Position[tab,Max[tab]]//First)+{0,cx-21}


TextRecognition[image_]:=(
   letterMaps=Chop[Map[probF[MVCorrelateImage[image,#[[2]],NormalizedSquaredEuclideanDistance]]&,letterTemplates]];
   letterMaps=Map[ImageData[MaxFilter[Image[#],{2,0}]]&,letterMaps];
   {width,height}={Length[letterMaps[[1,1]]],Length[letterMaps[[1]]]};

   \[Tau]1=Table[If[x1>width,0.,letterMaps[[l1,y,x1]]],{y,1,height},{l1,1,LettersLength},{x1,1,width+20}];
   \[Psi]1=\[Psi]Max[\[Tau]1];
   \[Phi]1=Table[\[Tau]1[[y,l1,x1]]*letterFrequencies[[l1]],{y,1,height},{l1,1,LettersLength},{x1,1,width}];

   \[Tau]2=Table[If[x2>width,0,letterMaps[[l2,y,x2]]*\[Psi]1[[l2,y,x2]]],{y,1,height},{l2,1,LettersLength},{x2,1,width+20}];
   \[Psi]2=\[Psi]Max[\[Tau]2];
   \[Phi]2=Table[\[Tau]2[[y,l2,x2]]*letterFrequencies[[l2]],{y,1,height},{l2,1,LettersLength},{x2,1,width}];

   \[Tau]3=Table[If[x3>width,0,letterMaps[[l3,y,x3]]*\[Psi]2[[l3,y,x3]]],{y,1,height},{l3,1,LettersLength},{x3,1,width+20}];
   \[Psi]3=\[Psi]Max[\[Tau]3];
   \[Phi]3=Table[\[Tau]3[[y,l3,x3]]*letterFrequencies[[l3]],{y,1,height},{l3,1,LettersLength},{x3,1,width}];

   \[Tau]4=Table[If[x4>width,0,letterMaps[[l4,y,x4]]*\[Psi]3[[l4,y,x4]]],{y,1,height},{l4,1,LettersLength},{x4,1,width+20}];
   \[Psi]4=\[Psi]Max[\[Tau]4];
   \[Phi]4=Table[\[Tau]4[[y,l4,x4]]*letterFrequencies[[l4]],{y,1,height},{l4,1,LettersLength},{x4,1,width}];

   \[Tau]5[y_,l5_,x5_]:=If[x5>width,0,letterMaps[[l5,y,x5]]*\[Psi]4[[l5,y,x5]]];
   \[Phi]5=Table[\[Tau]5[y,l5,x5]*letterFrequencies[[l5]],{y,1,height},{l5,1,LettersLength},{x5,1,width}];

   NumberOfLetters=(Ordering[{1,Max[\[Phi]1],Max[\[Phi]2],Max[\[Phi]3],Max[\[Phi]4],Max[\[Phi]5]},-1]//First)-1;
   letters=Switch[NumberOfLetters,
      0,{},
      1,Backtrack1[],
      2,Backtrack2[],
      3,Backtrack3[],
      4,Backtrack4[],
      5,Backtrack5[],
      _,Assert[1==2]]
)


TextRecognitionOutput[image_]:=(
   letters=TextRecognition[image];
   DispText[yb,letters,image]
)


TextRecognitionOutput1[image_]:=(
   letterFrequencies[[All]]=1;
   letterConditionalProbs[[All,All]]=1.;
   TextRecognitionOutput[image]
)


TextRecognitionOutput2[image_]:=(
   letterFrequencies=aletterFrequencies;
   letterConditionalProbs=aletterConditionalProbs;
   TextRecognitionOutput[image]
)


RandomImageLine[]:=(
   lineY=RandomInteger[{26,1496-26}];
   lineX=RandomInteger[{1,2048-101}];
   article[[lineY-25;;lineY+25,lineX;;lineX+100]])


Comp[image_]:={TextRecognitionOutput1[image],TextRecognitionOutput2[image]}


lines=Import["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\TextFragments.wdx"];


speak[]:=Speak[StringJoin[Map[letterTemplates[[#,1]]&,letters[[All,1]]]]]
