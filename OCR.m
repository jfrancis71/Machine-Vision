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
   {"i",article[[1018-7;;1018+10,224-4;;224+5]]},
   {"o",article[[1019-6;;1019+5,245-5;;245+5]]},
   {"v",article[[1019-6;;1019+5,307-5;;307+5]]},
   {"r",article[[1019-6;;1019+5,328-5;;326+5]]}
};


sourceText=Take[Characters[ExampleData[{"Text","OriginOfSpecies"}]],200];


(*Note I' m using the Laplacian adjustment*)
letterFrequencies=Table[(1+Count[sourceText,letterTemplates[[l,1]]])/(13+Sum[Count[sourceText,letterTemplates[[l,1]]],{l,1,12}]),{l,1,12}]//N;


freq=Table[Count[Map[#[[2]]&,Select[Partition[sourceText,2],#[[1]]==l1&]],l2],{l1,letterTemplates[[All,1]]},{l2,letterTemplates[[All,1]]}];


letterConditionalProbs=Table[(1+freq[[l1,l2]])/(12+Total[freq[[l1]]]),{l1,1,12},{l2,1,12}]//N;


probF[z_]=FullSimplify[PDF[HalfNormalDistribution[15],z],Assumptions->z>0];


letterWidths=Map[Dimensions[#[[2]]][[2]]&,letterTemplates]


MaxUnigram[image_]:=(
   letterFeatures=Map[MVCorrelateImage[image,#[[2]],NormalizedSquaredEuclideanDistance]&,letterTemplates];
   letterMaps=Map[probF[MVCorrelateImage[image,#[[2]],NormalizedSquaredEuclideanDistance]]&,letterTemplates];
   {width,height}={Length[letterMaps[[1,1]]],Length[letterMaps[[1]]]};
   \[Psi]1=Table[
      letterMaps[[l1,y,x1]],
      {l1,1,12},{y,1,height},{x1,1,width}];
   yb=Ordering[Table[Max[Table[\[Psi]1[[l1,y,x1]]*letterFrequencies[[l1]],{l1,1,12},{x1,1,width}]],
      {y,1,height}],-1]//First;
   x1b=Ordering[Table[Max[Table[\[Psi]1[[l1,yb,x1]]*letterFrequencies[[l1]],{l1,1,12}]],
      {x1,1,width}],-1]//First;
   l1b=Ordering[
      Table[\[Psi]1[[l1,yb,x1b]]*letterFrequencies[[l1]],{l1,1,12}],-1]//First;
   pTrue=Table[Max[Table[\[Psi]1[[l1,y,x1]]*letterFrequencies[[l1]],{l1,1,12},{x1,1,width}]],
      {y,1,height}]//Max;
   pFalse=PDF[NormalDistribution[0.5,0.15],letterFeatures[[l1b,yb,x1b]]];
   pTrue/(pTrue+pFalse)
)


MaxDigram[image_]:=(
   letterFeatures=Map[MVCorrelateImage[image,#[[2]],NormalizedSquaredEuclideanDistance]&,letterTemplates];
   letterMaps=Map[probF[MVCorrelateImage[image,#[[2]],NormalizedSquaredEuclideanDistance]]&,letterTemplates];
   {width,height}={Length[letterMaps[[1,1]]],Length[letterMaps[[1]]]};
   \[Psi]1=Table[
      letterMaps[[l1,y,x1]],
      {l1,1,12},{y,1,height},{x1,1,width}];
   \[Psi]2=Table[
      letterMaps[[l2,y,x2]]*
      Max[Table[letterConditionalProbs[[l2,l1]]*
      Exp[-((x1-x2)-(letterWidths[[l2]]+letterWidths[[l1]])/2)^2]*
      \[Psi]1[[l1,y,x1]]*If[x2>x1,0,1],{l1,1,12},{x1,x2,Min[width,x2+15]}]],
      {l2,1,12},{y,1,height},{x2,1,width}];
   yb=Ordering[Table[Max[Table[\[Psi]2[[l2,y,x2]]*letterFrequencies[[l2]],{l2,1,12},{x2,1,width}]],
      {y,1,height}],-1]//First;
   x2b=Ordering[Table[Max[Table[\[Psi]2[[l2,yb,x2]]*letterFrequencies[[l2]],{l2,1,12}]],
      {x2,1,width}],-1]//First;
   l2b=Ordering[
      Table[\[Psi]2[[l2,yb,x2b]]*letterFrequencies[[l2]],{l2,1,12}],-1]//First;
   l1b=Ordering[Table[
      Max[Table[letterConditionalProbs[[l2b,l1]]*
      Exp[-((x1-x2b)-(letterWidths[[l2b]]+letterWidths[[l1]])/2)^2]*
      \[Psi]1[[l1,yb,x1]]*If[x2b>x1,0,1],{x1,1,width}]],
      {l1,1,12}],-1]//First;
   x1b=Ordering[Table[letterConditionalProbs[[l2b,l1b]]*
      Exp[-((x1-x2b)-(letterWidths[[l2b]]+letterWidths[[l1b]])/2)^2]*
      \[Psi]1[[l1b,yb,x1]]*If[x2b>x1,0,1],{x1,1,width}],-1]//First;
   pTrue=Table[Max[Table[\[Psi]2[[l2,y,x2]]*letterFrequencies[[l2]],{l2,1,12},{x2,1,width}]],
      {y,1,height}]//Max;
   pFalse=PDF[NormalDistribution[0.5,0.15],letterFeatures[[l1b,yb,x1b]]]*
      PDF[NormalDistribution[0.5,0.15],letterFeatures[[l2b,yb,x2b]]];
   pTrue/(pTrue+pFalse)
)
