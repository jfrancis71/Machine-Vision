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
