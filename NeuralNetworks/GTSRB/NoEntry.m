(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


NNRead["GTSRB\\GTSRB"]


MyNew=Delete[wl,{{1},{5},{9}}];


GetPatch[image_,coords_]:=image[[coords[[2]]-16;;coords[[2]]+15,coords[[1]]-16;;coords[[1]]+15]]


PlotNoEntry[image_?MatrixQ]:=(proc1=ForwardPropogate[{image},MyNew[[1;;-4]]];
final=Table[ForwardPropogate[Flatten[proc1[[1,All,yp;;yp+4-1,xp;;xp+4-1]]].MyNew[[-2,2,1]]+-7.503736,MyNew[[-1;;-1]]],{yp,1,Length[proc1[[1,1]]]-4},{xp,1,Length[proc1[[1,1,1]]]-4}];
pos=Position[final,q_/;q>.000001];
npos=Map[({#[[1]],#[[2]]}-{1,1})*8+{14,14}&,pos];
cpos=Map[{16+#[[2]],16+#[[1]]}&,npos];
zpos=Select[cpos,ForwardPropogate[{GetPatch[image,#]},wl][[1,1]]>.999&];
Show[image//DispImage,MapThread[OutlineGraphics[BoundingRectangles[{#1},{16,16}],Blend[{Pink,Blue},#2]]&,{zpos,zpos}]])


imgy1=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0772.JPG",385];


imgy2=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0773.JPG",460];


imgy3=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0774.JPG",200];


imgy4=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0775.JPG",460];


imgy5=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0776.JPG",400];


imgy6=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0777.JPG",460];


imgy7=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0778.JPG",190];


imgy8=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0779.JPG",200];


imgy9=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0780.JPG",320];


imgy10=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0781.JPG",320];


imgy11=StandardiseImage["C:\\Users\\julian\\Google Drive\\Personal\\Pictures\\Vision Experiments\\Traffic signs\\IMG_0782.JPG",360];


Demo[]:=(
   pl1=PlotNoEntry[imgy1];
   pl2=PlotNoEntry[imgy2];
   pl3=PlotNoEntry[imgy3];
   {pl1,pl2,pl3}
(* Picks out correctly for the 3 images.
   This is from the 4'th iteration of the GTSRB training algorithm *)
)
