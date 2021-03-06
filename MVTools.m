(* ::Package:: *)

(* Returned structure is in raster addressing convention
   ie first row corresponds to bottom row in image.
   So do not use directly in Array/MatrixPlot
*)
StandardiseImage[image_Image,size_Integer]:=ImageResize[ColorConvert[image,"GrayScale"],size]//ImageData//Reverse
StandardiseImage[image_Image,{sy_Integer,sx_Integer}]:=ImageResize[ColorConvert[image,"GrayScale"],{sy,sx}]//ImageData//Reverse
StandardiseImage[image_Image]:=StandardiseImage[image,128]
StandardiseImage[image_String, size_Integer] := StandardiseImage[Import[image],size]
StandardiseImage[image_String, {sy_Integer,sx_Integer}] := StandardiseImage[Import[image],{sy,sx}]
StandardiseImage[image_String] := 
   StandardiseImage[Import[image]];
StandardiseImage[image_?MatrixQ,size_Integer]:=ImageData[ImageResize[Image[image],size]]

ReadImagesFromDirectory[directory_String,size_:128]:=
   Map[StandardiseImage[#,size]&,
   FileNames[StringJoin[directory,"\\*.jpg"]]
]


ReadMovie[fileName_String,startFrame_Integer,endFrame_Integer,width_Integer:128]:=
   Table[StandardiseImage[Import[fileName,{"Frames",{t}}],width],{t,startFrame,endFrame}];


ReadMovieLength[fileName_String]:=Length[Import[fileName]];


DispImage[image_?MatrixQ]:={image//Raster,Red,If[Length[image]<32,Point[{Length[image],Length[image]}/2],{}]}//Graphics;

ColDispImage[image_]:=Raster[Map[List@@Blend[{Blue,Red},#/2+.5]&,image,{2}]]//Graphics;

MVImageFilter[f_,image_?MatrixQ,kernelSize_]:=ImageFilter[f,image//Image,kernelSize]//ImageData;
MVPyramidFilter[f_,pyr_?PyramidImageQ,kernelSize_]:=Map[MVImageFilter[f,#,kernelSize]&,pyr];


(* Builds an image pyramid.
   Stops before image is smaller than the filter. FilterSize={rows,cols}
   Note filterSize is given in {r,c} pixels extending from the centre
   so filterDims=filterSize*2 + {1,1}
*)
BuildPyramid[image_?MatrixQ,filterSize_List:{8,8}]:=
   Table[ImageResize[Image[image],Round[Dimensions[image][[2]]*(1-.1)^n]]//ImageData,{n,0,-Log[Min[Dimensions[image]]/(2*filterSize[[2]]+1)]/Log[1-.1]}]
   PyramidImageQ[image_]:=ArrayQ[image,3]

PyramidImageQ[pyramid_List]:=MatrixQ[pyramid[[1]]]

PyramidFilter[f_,pyr_?PyramidImageQ,r_List]:=Map[(ImageFilter[f,#//Image,r]//ImageData)&,pyr];

(* Note this doesn't assume filter *)
InBoundsQ[image_?MatrixQ,y_,x_] := (y>0&&x>0&&y<Length[image]&&x<Length[image[[1]]])

(* Note this does assume filter *)
InBoundsQ[pyr_?PyramidImageQ,level_,y_,x_,filterSize_:8] := (level>0&&Length[pyr]>level&&y>filterSize&&x>filterSize&&y<Length[pyr[[level]]]-filterSize&&x<Length[pyr[[level,1]]]-filterSize)


Patch[image_?MatrixQ,y_,x_,filterSize_:8]:=image[[y-filterSize;;y+filterSize,x-filterSize;;x+filterSize]];
Patch[pyr_,level_,y_,x_,filterSize_:8]:=Patch[pyr[[level]],y,x,filterSize];

DrawKernelOnPatch[patch_,kernel_]:=Show[patch//DispImage,Graphics[{Red,Map[Point[Reverse[#-{.5,.5}]]&,Position[kernel,1.]]}]]


MVCorrelateImage[image_?MatrixQ,kernel_?MatrixQ,f_:Dot]:=ImageData[ImageCorrelate[Image[image],kernel,f]];
MVCorrelatePyramid[pyramid_?PyramidImageQ,kernel_?MatrixQ,f_:Dot]:=Map[MVCorrelateImage[#,kernel,f]&,pyramid]


(* Von Mises Distribution
   kernel is a masking kernel to indicate some angle not relevant. *)
EdgeDirConv[edgePyramid_,kernel_,angles_]:=
   MVCorrelatePyramid[Cos[edgePyramid], kernel*Cos[angles]] +
      MVCorrelatePyramid[Sin[edgePyramid],kernel*Sin[angles]]


sobelX = {{-1, 0, +1}, {-2, 0, +2}, {-1, 0, +1}}; sobelY = 
 Transpose[sobelX];
EdgeMag[dx_,dy_]:=Sqrt[dx^2+dy^2];(*SetAttributes[EdgeMag,Listable];*)

(* Returns direction of maximum light increase *)
EdgeDir[dx_,dy_]=ArcTan[dx,dy];
EdgeDir[0.0,0.0]=0.0;SetAttributes[EdgeDir,Listable];

(* Returns angle in direction of surface, no polarity *)
SurfDir[0,0]:=0.
SurfDir[dx_,dy_]:=Mod[ArcTan[dx,dy]+\[Pi]/2//N,\[Pi]]
SurfDir[0.0,0.0]:=0.0

EdgeMagDir[dx_,dy_]:=Transpose[{EdgeMag[dx,dy],EdgeDir[dx,dy]},{3,1,2}]

SurfMagDir[dx_,dy_]:=Transpose[{EdgeMag[dx,dy],SurfDir[dx,dy]},{3,1,2}]

SobelFilter[image_?MatrixQ,func_]:=func[
   MVCorrelateImage[image,sobelX],
   MVCorrelateImage[image,sobelY]
]

SobelFilter[pyramid_?PyramidImageQ,func_]:=
   Map[SobelFilter[#,func]&,pyramid]


EdgeMagDirToSurfDir[edgeMagDir_]:=Mod[edgeMagDir[[All,All,All,2]]+\[Pi]/2,\[Pi]];
EdgeMagDirToEdgeDir[edgeMagDir_]:=edgeMagDir[[All,All,All,2]];
EdgeMagDirToEdgeMag[edgeMagDir_]:=edgeMagDir[[All,All,All,1]];


BoundingRectangles[coords_?MatrixQ,filterSize_?VectorQ]:=
   Map[Rectangle[#-filterSize,#+filterSize]&,coords]

BoundingRectangles[imgPyramid_?PyramidImageQ,coords_?MatrixQ,filterSize_?VectorQ]:=
   Map[Scale[BoundingRectangles[{#[[2;;3]]//Reverse},filterSize],Length[imgPyramid[[1]]]/Length[imgPyramid[[#[[1]]]]],{0,0}]&,coords]

BoundingRectangles[imgPyramid_?PyramidImageQ,{},filterSize_?VectorQ]:={}

(* Pyramid is assumed to be of some filter type which has already been applied. Positives are values above the threshold *)
BoundingRectangles[pyramid_?PyramidImageQ,threshold_Real,filterSize_?VectorQ]:=
   BoundingRectangles[pyramid,Position[pyramid,x_/;x>=threshold],filterSize]

OutlineGraphics[grObjects_,Colour_:Green]:=Graphics[{Opacity[0],Colour,EdgeForm[Directive[Colour,Thick]],grObjects}]


(* image is the background image, not necessarily level 1 of pyramid, esp if pyramid is edge pyramid
   displayFunc is the function to display the extracted subwindow in the pyramid *)

(* len,dims implementation issue, seems to not like evaluating length inside the manipulate expression *)
PyramidExplorer[image_?MatrixQ,pyramid_?PyramidImageQ,displayFunc_,startPoint_:{7,100,100}]:=(
   len=Length[pyramid];dims=Table[Dimensions[pyramid[[l]]],{l,1,len}];
   Manipulate[(
      y=Min[Max[9,1+Floor[-.00001+ry*dims[[l,1]]]],1+dims[[l,1]]-9];x=Min[Max[9,x=1+Floor[-.00001+rx*dims[[l,2]]]],1+dims[[l,2]]-9]);
      {
      Show[image//DispImage,OutlineGraphics[BoundingRectangles[pyramid,{{l,y,x}},{8,8}]]],
      {rx,ry},{x,y},
      pyramid[[l,y-8;;y+8,x-8;;x+8]]//displayFunc
      },{{l,startPoint[[1]]},1,len,1},{{ry,startPoint[[2]]/dims[[startPoint[[1]],1]]//N},0,1},{{rx,startPoint[[3]]/dims[[startPoint[[1]],2]]//N},0,1}])

PyramidExplorer[pyramid_?PyramidImageQ,startPoint_:{7,100,100}]:=PyramidExplorer[pyramid[[1]],pyramid,DispImage,startPoint]

DrawSurfDirCell[centre_?VectorQ,value_]:=
   {Black,Line[{centre-0.5*{Cos[value],Sin[value]},centre+0.5*{Cos[value],Sin[value]}}]}

DrawEdgeDirCell[centre_?VectorQ,value_]:={
   Black,Line[{centre-0.4*{Cos[value],Sin[value]},centre+0.4*{Cos[value],Sin[value]}}],
   Blue,Point[{centre+0.4*{Cos[value],Sin[value]}}] }


DrawEdgeMagDirCell[centre_?VectorQ,{mag_,dir_}]:={
   RGBColor[0,mag,0],Line[{centre-0.4*{Cos[dir],Sin[dir]},centre+0.4*{Cos[dir],Sin[dir]}}],
   Blue,Point[{centre+0.4*{Cos[dir],Sin[dir]}}] };


DrawSurfMagDirCell[centre_?VectorQ,{mag_,dir_}]:={
   RGBColor[0,mag,0],Line[{centre-0.4*{Cos[dir],Sin[dir]},centre+0.4*{Cos[dir],Sin[dir]}}] };


(* Used for plotting grid of cell values
   eg. CellPlot[patch,DrawEdgeMagDirCell]
*)
CellPlot[patch_,cellPlotF_]:=Graphics[Join[{Red,Point[{8.5,8.5}]},Flatten[Table[cellPlotF[{x+8+0.5,y+8+0.5},patch[[y+9,x+9]]],{y,-8,+8},{x,-8,+8}],1]]]


FilterAnalyse[filterPyramid_?PyramidImageQ,patchFilterF_,kernel_,ChannelDrawF_,coords_?VectorQ]:=
   {Show[kernel//DispImage,filterPyramid[[coords[[1]],coords[[2]]-8;;coords[[2]]+8,coords[[3]]-8;;coords[[3]]+8]]//ChannelDrawF],filterPyramid[[coords[[1]],coords[[2]]-8;;coords[[2]]+8,coords[[3]]-8;;coords[[3]]+8]]//patchFilterF//Reverse//MatrixPlot}

(* Example Use:
   FilterAnalyse[surfPyr,SurfCirclePatchH,circleKernel,SurfDirPlot,{7,130,183}]
*)


LogSum[a_,b_]:=Max[a,b]+Log[Exp[a-Max[a,b]]+Exp[b-Max[a,b]]]


MVProfile[f_,expon_]:=Print["Timings: ",AbsoluteTiming[
   Table[f,{10^expon}];][[1]]*10^(6-expon)," microseconds"];
Timer[message_,expr_]:=Module[{timer=AbsoluteTiming[expr]},If[MatchQ[$timers,True],Print[message," ",timer[[1]]," secs"]];timer[[2]]];
SetAttributes[Timer,HoldAll];
SetAttributes[MVProfile,HoldFirst]


CameraRecognition[program_,imageWidth_:128]:=(
   Print[Dynamic[out]];
   While[True,PreemptProtect[
      out=program[currentImg=StandardiseImage[CurrentImage[],imageWidth]]]]);
MobileRecognition[program_,stationNo_Integer,imageWidth_:128]:=(
   Print[Dynamic[out]];
   While[True,PreemptProtect[
      out=program[currentImg=StandardiseImage[Import["http://192.168.0."<>ToString[stationNo]<>"/image.jpg"],imageWidth]]]]);


(* http://www.cs.toronto.edu/~fritz/absps/momentum.pdf *)
(* On the importance of initialization and momentum in deep learning *)
(* Sutskever, Martens, Dahl, Hinton (2013) *)
SyntaxInformation[Momentum]={"ArgumentsPattern"->{}};
SyntaxInformation[MomentumType]={"ArgumentsPattern"->{}};
SyntaxInformation[LearningRate]={"ArgumentsPattern"->{}};
SyntaxInformation[MaxEpoch]={"ArgumentsPattern"->{}};
SyntaxInformation[IterationMonitor]={"ArgumentsPattern"->{}};
SyntaxInformation[EpochMonitor]={"ArgumentsPattern"->{}};
SyntaxInformation[IterationFunctions]={"ArgumentsPattern"->{}};
NullFunction[]=Function[{},(Null)];
Options[GradientDescent]={
   LearningRate->.01,
   MaxEpoch->10,
   Momentum->0.0,
   MomentumType->"CM",
   EpochMonitor->NullFunction, (* Function to be evaluated at end of each epoch *)
   IterationFunctions->{} (* List of functions to be evaluated as iteration proceeds.
      Each is passed current state, ie f(x) type of calculation *)
};

GradientDescent[initialState_,gradF_,plusF_,opts:OptionsPattern[]]:=
   GradientDescent[initialState,{gradF},plusF,opts];

GradientDescent[initialState_,gradFunctions_List,plusF_,opts:OptionsPattern[]]:=(
   init = 0;velocity=0.0;
   For[state=initialState;epoch=0,epoch<=OptionValue[MaxEpoch],epoch++,
      For[iter=1,iter<=Length[gradFunctions],iter++,
         gw=If[OptionValue[MomentumType]!="Nesterov"||init==0,
            gradFunctions[[iter]][state],
            gradFunctions[[iter]][plusF[state,OptionValue[Momentum]*velocity]]];
         velocity = OptionValue[Momentum]*velocity + -OptionValue[LearningRate]*gw; init=1;
         state = plusF[state,velocity];
         If[Length[OptionValue[IterationFunctions]]>0,OptionValue[IterationFunctions][[iter]][state]];
         ];
      OptionValue[EpochMonitor][];
      ];
   state);


(*
   Alexey Popkov - From Stackoverflow 17 Aug 2011
   http://stackoverflow.com/questions/3418892/profiling-memory-usage-in-mathematica
   http://stackoverflow.com/users/590388/alexey-popkov
*)
myByteCount[symbolName_String]:=Replace[ToHeldExpression[symbolName],Hold[x__]:>If[MemberQ[Attributes[x],Protected|ReadProtected],Sequence@@{},{ByteCount[Through[{OwnValues,DownValues,UpValues,SubValues,DefaultValues,FormatValues,NValues}[Unevaluated@x,Sort->False]]],symbolName}]];

MemoryReport[]:=With[{listing=myByteCount/@Names[]},Labeled[Grid[Reverse@Take[Sort[listing],-100],Frame->True,Alignment->Left],Column[{Style["ByteCount for symbols without attributes Protected and ReadProtected in all contexts",16,FontFamily->"Times"],Style[Row@{"Total: ",Total[listing[[All,1]]]," bytes for ",Length[listing]," symbols"},Bold]},Center,1.5],Top]]
(*End of attribution*)


On[Assert];
Test1=Table[Random[],{i,1,6},{j,1,6}];
Assert[Dimensions[StandardiseImage[Test1//Image,36]][[2]]==36];

(TestB=BoxMatrix[1,5]);
