(* ::Package:: *)

PlotNNFaceRecognition[image_]:=( 
(* We don't convolve with whole image. Issue appears regarding the padding around our filter operations *)

   partitionImage=Partition[image,{32,32},{4,4}];
   flattenImage=Flatten[partitionImage,1];
   processed=ForwardPropogate[flattenImage,wl][[All,1]];
   tableProc=Partition[processed,Length[partitionImage[[1]]]];
   pos=Map[Reverse,Position[tableProc,z_/;z>.9]];
   centres=Map[(#-{1,1})*4+{16,16}&,pos];
   Show[image//DispImage,OutlineGraphics[BoundingRectangles[centres,{16,16}]]]
)


FaceDetection[image_]:=(
   rey=ImageData[ImageResize[image//Image,43]][[All,6;;-7]];
   rec=ForwardPropogate[{ImageData[ImageResize[image//Image,43]][[All,6;;-7]]},wl];
   {rey//DispImage,BarChart[rec,PlotRange->{0,1}]}
)


rectify[MathImage_]:=Partition[ColorConvert[ImageResize[MathImage,Scaled[32/Min[ImageDimensions[MathImage]]]],"GrayScale"]//ImageData//Reverse,{32,32},{1,1}]


score[MathImage_]:=Max[ForwardPropogate[Flatten[rectify[MathImage],1],wl]];


WebImportImages[url_]:=Import[url,"Images"];


(* Quite primitive, will require that the scaling for the image is appropriate or
   altering the scale factor below
*)
FaceMap[image_]:=(
   im=ImageData[ColorConvert[ImageResize[image,{2592,1944}/16],"GrayScale"]]//Reverse;
   parts=Partition[im,{32,32},{1,1}];
   lineDens=ForwardPropogate[parts[[50]],wl][[All,1]];
   dens=Monitor[Table[ForwardPropogate[parts[[l]],wl][[All,1]],{l,1,91}],l];
)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/Dream.m"


NNRead["FaceScrub\\GenderNet"];GenderNet=wl;
NNRead["FaceScrub\\FaceScrub"];


FaceUI[mirror_Symbol:True]:=CameraRecognition[
   Function[image,{
      "                ",
      (im1=Map[If[mirror,Reverse,#&],image[[64-31;;64+31;;2,64-31;;64+31;;2]]]);
      gender=ForwardPropogate[{im1},GenderNet][[1,1]];
      BarChart[ForwardPropogate[{im1},wl],PlotRange->{0,1},ChartStyle->Blend[{Pink,Blue},gender]],im1//DispImage,Abs[Salient[{1},im1]]//DispImage}]];


GetPatch[image_,coords_]:=image[[coords[[2]]-16;;coords[[2]]+15,coords[[1]]-16;;coords[[1]]+15]]


(* Note, padding the image at beginning is not the same as padding arrays at each stage of neural net pipeline. Difference is biases of neurones.
Secondly, just doing convolution on whole image is not equivelant to either of above. Because the patch is incorporating information from outside the filter field. This is neither the same as zero's nor biases.
Could try learning with L1 neural activity. Have no idea if this is helpful or not.
Manually reset bias?
*)
MyNew=Delete[wl,{{1},{5},{9}}];
PlotFace[image_?MatrixQ]:=(
   proc1=ForwardPropogate[{image},MyNew[[1;;-4]]];
   final=Table[ForwardPropogate[Flatten[proc1[[1,All,yp;;yp+4-1,xp;;xp+4-1]]].MyNew[[-2,2,1]]+-7.503736,MyNew[[-1;;-1]]],{yp,1,Length[proc1[[1,1]]]-4},{xp,1,Length[proc1[[1,1,1]]]-4}];
   pos=Position[final,q_/;q>.5];
   npos=Map[({#[[1]],#[[2]]}-{1,1})*8+{14,14}&,pos];
   cpos=Map[{16+#[[2]],16+#[[1]]}&,npos];
   zpos=Select[cpos,ForwardPropogate[{GetPatch[image,#]},wl][[1,1]]>.5&];
   genders=If[Length[zpos]>=1,ForwardPropogate[Map[GetPatch[image,#]&,zpos],GenderNet][[All,1]],{}];
   Show[image//DispImage,MapThread[OutlineGraphics[BoundingRectangles[{#1},{16,16}],Blend[{Pink,Blue},#2]]&,{zpos,genders}]]
)
