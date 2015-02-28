(* ::Package:: *)

InsideBar[{x_,y_}]:=Abs[y]<1.5&&Abs[x]<=6.5;
InsideBar[struct_]:=Boole[Map[Abs[#[[2]]]<1.5&&Abs[#[[1]]]<=6.5&,struct,{3}]];
InsideBar[struct_]:=(1.-UnitStep[Abs[struct[[All,All,All,2]]]-1.5])*(1.-UnitStep[Abs[struct[[All,All,All,1]]]-6.5]);
InsideBar[struct_,barl_,barr_,barb_,bart_]:=UnitStep[struct[[All,All,All,1]]-barl]*UnitStep[barr-struct[[All,All,All,1]]]*
   UnitStep[struct[[All,All,All,2]]-barb]*UnitStep[bart-struct[[All,All,All,2]]]


InsideInnerSign[{x_,y_}]:=Norm[{x,y}]<=7.5&&!InsideBar[{x,y}];
InsideInnerSign[struct_]:=Boole[Map[Norm[#]<=7.5&&!InsideBar[#]&,struct,{3}]];
InsideInnerSign[struct_]:=(1-UnitStep[Map[Norm,struct,{3}]-7.5])*(1-InsideBar[struct]);
InsideInnerSign[struct_,posX_,posY_,scale_,barl_,barr_,barb_,bart_]:=(1-UnitStep[Map[Norm[#-{posX,posY}]&,struct,{3}]-7.5*scale])*(1-InsideBar[struct,barl,barr,barb,bart])


InsideOuterRing[{x_,y_}]:=7.5<Norm[{x,y}]<7.8;
InsideOuterRing[struct_]:=Map[Boole[7.5<Norm[#]<7.8]&,struct,{3}];
InsideOuterRing[struct_,posX_,posY_,scale_]:=Map[Boole[7.5*scale<Norm[#-{posX,posY}]<7.8*scale]&,struct,{3}]


InsideNoEntrySign[{x_,y_}]:=Norm[{x,y}]<7.8;
InsideNoEntrySign[struct_]:=Map[Boole[Norm[#]<7.8]&,struct,{3}];
InsideNoEntrySign[struct_,posX_,posY_,scale_]:=Map[Boole[Norm[#-{posX,posY}]<7.8*scale]&,struct,{3}]


Map[DensityPlot[Boole[#[{x,y}]],{x,-8,+8},{y,-8,+8}]&,{InsideBar,InsideInnerSign,InsideOuterRing,InsideNoEntrySign}]


ptchMonte=Table[{Random[],Random[]}-{0.5,0.5},{40}];


transform[{px_,py_},{s_,mx_,my_}]:={
Total[Map[1-Boole[InsideNoEntrySign[({px,py}+#+{mx,my})*s]]&,ptchMonte]]/40.,
Total[Map[Boole[InsideInnerSign[({px,py}+#+{mx,my})*s]]&,ptchMonte]]/40.,
Total[Map[Boole[InsideOuterRing[({px,py}+#+{mx,my})*s]]+Boole[InsideBar[({px,py}+#+{mx,my})*s]]&,ptchMonte]]/40.
}


pyrs=Map[BuildPyramid,positives];


candList=Select[Import["C:\\Users\\Julian\\secure\\Shape Recognition\\Stop Sign\\Positions.wdx"],InBoundsQ[pyrs[[1]],#[[2]],#[[3]],#[[4]]]&];


patches=Map[Patch[pyrs[[#[[1]]]],#[[2]],#[[3]],#[[4]]]&,candList];


w=Integrate[PDF[UniformDistribution[{0,f1}],x-z] PDF[NormalDistribution[.3 f2 b1+ .95 f3 b1,Sqrt[.1^2 f2^2 + .1^2 f3^2]],z],
{z,-\[Infinity],+\[Infinity]}]


probDens[v_,{f1_,f2_,f3_}]:=(1/f1)0.5` (-1.` Erf[(0.7071067811865475` (8. f2+19. f3-20. v))/Sqrt[f2^2+f3^2]]+Erf[1/(Sqrt[f2^2+f3^2])0.7071067811865475` (20. f1+8.f2+19.` f3-20.v)])


probDens[v_,{f1_,f2_,f3_},br_]=(w /. x->v /. b1->br);


LogSum[l1_,l2_]:=Max[l1,l2]+Log[Exp[l1-Max[l1,l2]]+Exp[l2-Max[l1,l2]]]


toDisk:={{1,0,0},{0,1,1},{0,0,0}};


pointCloud=Table[{x,y}+ptchMonte[[m]],{y,-8,+8},{x,-8,+8},{m,1,40}];
pointCloud//Length


NoEntrySignPatch[patch_,b_,{s_,mx_,my_}]:=
Log[Product[Max[probDens[patch[[y+9,x+9]],transform[{x,y},{s,mx,my}]+{1,1,1}*.01,b],.001],{y,-8,+8},{x,-8,8}]]-
LogSum[3.1 \[Pi] s^2,Log[Product[Max[probDens[patch[[y+9,x+9]],toDisk.transform[{x,y},{s,mx,my}]+{1,1,1}*.01,b],.001],{y,-8,+8},{x,-8,8}]]]


num[patch_,b_,{s_,mx_,my_}]:=( 
def1=Table[(pointCloud[[y+9,x+9]]+ConstantArray[{mx,my},40])*s,{y,-8,+8},{x,-8,+8}];
shp2=Transpose[{
40-Total[InsideNoEntrySign[def1],{3}],
Total[InsideInnerSign[def1],{3}],
Total[InsideOuterRing[def1]+InsideBar[def1],{3}]
}/40,{3,1,2}];
Log[Table[Max[probDens[patch[[y+9,x+9]],shp2[[y+9,x+9]]+{1,1,1}*.01,b],.001],{y,-8,+8},{x,-8,8}]])


NoEntrySignPatch[patch_,b_,{barl_,barr_,barb_,bart_,posX_,posY_,scale_}]:=
( 
   def1=Table[pointCloud[[y+9,x+9]],{y,-8,+8},{x,-8,+8}];
   shp2=Transpose[{
      40-Total[InsideNoEntrySign[def1,posX,posY,scale],{3}],
      Total[InsideInnerSign[def1,posX,posY,scale,barl,barr,barb,bart],{3}],
      Total[InsideOuterRing[def1,posX,posY,scale]+InsideBar[def1,barl,barr,barb,bart],{3}]
   }/40,{3,1,2}];
   ntab=Table[Log[Max[probDens[patch[[y+9,x+9]],shp2[[y+9,x+9]]+{1,1,1}*.001,b],.001]],{y,-8,+8},{x,-8,8}];
   (numer=(ntab//Flatten//Total))-(denom=LogSum[\[Pi] (7.8*scale)^2,(denomDisk=Table[Log[Max[probDens[patch[[y+9,x+9]],toDisk.shp2[[y+9,x+9]]+{1,1,1}*.01,b],.001]],{y,-8,+8},{x,-8,8}])//Flatten//Total]))


NoEntrySignPatch[patch_]:=.1*Sum[NoEntrySignPatch[patch,b,{s,mx,my}],{b,0.1,1,.1},{s,0.8,1.2,0.1},{mx,-2,2,0.5},{my,-.5,+.5,0.5}]


NoEntrySignPatchDiag[patch_]:=.1*Table[NoEntrySignPatch[patch,b,{s,mx,my}],{b,0.3,.45,.05},{s,0.8,1.3,0.1},{mx,-2,2,0.5},{my,-.5,+.5,0.5}]


NoEntrySignPatch[patch_]:=
.1*Sum[Exp[.1*
Sum[
If[InsideBar[{x,y},0],Log[1/(Sqrt[2 \[Pi]] .05)]-(patch[[y+9,x+9]]-0.95*b)^2/(2 .05^2),0]+
If[InsideInnerSign[{x,y},0],Log[1/(Sqrt[2 \[Pi]] .05)]-(patch[[y+9,x+9]]-0.4*b)^2/(2 .05^2),0]
,{y,-8,+8},{x,-8,+8}]],{b,0,1,0.1}]


FilterOutput[image_,pyr_,res1_]:=( 
rnk=20.;
focus=Position[res1,x_/;x>=rnk];
focus=Select[focus,InBoundsQ[pyr,#[[1]],#[[2]],#[[3]]]&];

res2=res1*.0;
res3=ReplacePart[res2,
Map[#->NoEntrySignPatch[Patch[pyr,#[[1]],#[[2]],#[[3]]]]&,focus]];
Show[image//DispImage,BoundingRectangles[res3,12000.,{8,8}]//OutlineGraphics]  )


NoEntryRecognitionOutput1[image_]:=( 
res=NoEntryRecognition[image];
FilterOutput[image])


b*.95


pt=patches[[23]];{
b=loc[[1]];(*b=.4;*)
s=loc[[8]];(*s=1.05;*)
mx=loc[[3]];(*mx=-1;*)
my=loc[[4]];
NoEntrySignPatch[pt,b,{s,mx,my}],
Show[{
pt[[1;;17,1;;17]]//DispImage,
OutlineGraphics[{Rectangle[{loc[[2]],loc[[4]]}+{8.5,8.5},{loc[[3]],loc[[5]]}+{8.5,8.5}]
}
],
{Green,
Circle[{8.5,8.5}+{mx,my},7.5*s],
Circle[{8.5,8.5}+{mx,my},7.8*s],
Point[{0,0}+{8.5,8.5}]}//Graphics
}],
NumberForm[(frMe=Log[Table[
Max[probDens[pt[[y+9,x+9]],transform[{x,y},{s,mx,my}]+{1,1,1}*.01,b],.001]
,{y,-8,+8},{x,-8,+8}]])//Reverse//MatrixForm,{3,3}],NumberForm[pt[[1;;17,1;;9]]//Reverse//MatrixForm,{3,3}]}


baseParams={0.5,-6.5,+6.5,-1.5,+1.5,0,0,1};


GradientShape1[patch_,{b_,barl_,barr_,barb_,bart_,posX_,posY_,scale_}]:=
(gr={
NoEntrySignPatch[patch,b+.01,{barl,barr,barb,bart,posX,posY,scale}]-
NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale}],
NoEntrySignPatch[patch,b,{barl+.2,barr,barb,bart,posX,posY,scale}]-
NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale}],
NoEntrySignPatch[patch,b,{barl,barr+.2,barb,bart,posX,posY,scale}]-
NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale}],
NoEntrySignPatch[patch,b,{barl,barr,barb+.2,bart,posX,posY,scale}]
-NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale}],
NoEntrySignPatch[patch,b,{barl,barr,barb,bart+.2,posX,posY,scale}]
-NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale}],
NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX+.2,posY,scale}]
-NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale}],
NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY+.2,scale}]
-NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale}],
NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale+.05}]
-NoEntrySignPatch[patch,b,{barl,barr,barb,bart,posX,posY,scale}]
}*{100,5,5,5,5,5,5,20})


GradientAscent[patch_,iter_:20]:=(For[t6=1;loc=baseParams,t6<=iter,t6=t6+1,(tmp=8;loc=0.01*Normalize[GradientShape1[patch,loc]]+loc;
r=4;
val=NoEntrySignPatch[patch,loc[[1]],loc[[2;;-1]]];shp4=shp2;shp5=ntab;
)
];val)
