(* ::Package:: *)

(* Returned structure is in raster addressing convention
   ie first row corresponds to bottom row in image.
   So do not use directly in Array/MatrixPlot
*)
StandardiseImage[image_Image,size_Integer]:=ImageResize[ColorConvert[image,"GrayScale"],size]//ImageData//Reverse
StandardiseImage[image_Image]:=StandardiseImage[image,128]
StandardiseImage[image_String, size_Integer] := 
   StandardiseImage[Import[image],size]
StandardiseImage[image_String] := 
   StandardiseImage[Import[image]]


DispImage[image_?MatrixQ]:=image//Raster//Graphics


(* Builds an image pyramid.
   Stops before image is smaller than the filter. FilterSize={rows,cols}
   Note filterSize is given in {r,c} pixels extending from the centre
   so filterDims=filterSize*2 + {1,1}
*)
BuildPyramid[image_?MatrixQ,filterSize_List]:=
   Table[ImageResize[Image[image],Round[Dimensions[image][[2]]*(1-.1)^n]]//ImageData,{n,0,-Log[Dimensions[image][[2]]/(2*filterSize[[2]]+1)]/Log[1-.1]}]
   PyramidImageQ[image_]:=ArrayQ[image,3]

PyramidImageQ[pyramid_List]:=MatrixQ[pyramid[[1]]]


MVCorrelateImage[image_?MatrixQ,kernel_?MatrixQ]:=ImageData[ImageCorrelate[Image[image],kernel]];
MVCorrelatePyramid[pyramid_?PyramidImageQ,kernel_?MatrixQ]:=Map[MVCorrelateImage[#,kernel]&,pyramid]


sobelX = {{-1, 0, +1}, {-2, 0, +2}, {-1, 0, +1}}; sobelY = 
 Transpose[sobelX];
EdgeMag[dx_,dy_]=Sqrt[dx^2+dy^2];SetAttributes[EdgeMag,Listable];

(* Returns direction of maximum light increase *)
EdgeDir[dx_,dy_]=ArcTan[dx,dy];
EdgeDir[0.0,0.0]=0.0;SetAttributes[EdgeDir,Listable];

(* Returns angle in direction of surface, no polarity *)
SurfDir[dx_,dy_]:=Mod[ArcTan[dx,dy]+\[Pi]/2//N,\[Pi]]
SurfDir[0.0,0.0]:=0.0

EdgeMagDir[dx_,dy_]:=Transpose[{EdgeMag[dx,dy],EdgeDir[dx,dy]},{3,1,2}]

SobelFilter[image_?MatrixQ,func_]:=func[
   MVCorrelateImage[image,sobelX],
   MVCorrelateImage[image,sobelY]
]

SobelFilter[pyramid_?PyramidImageQ,func_]:=
   Map[SobelFilter[#,func]&,pyramid]


BoundingRectangles[image_?MatrixQ,threshold_Real,filterSize_?VectorQ]:=
   Map[Rectangle[#-filterSize,#+filterSize]&,Reverse/@Position[image,x_/;x>=threshold]];

(* Pyramid is assumed to be of some filter type which has already been applied. Positives are values above the threshold *)
BoundingRectangles[pyramid_?PyramidImageQ,threshold_Real,filterSize_?VectorQ]:=
   Map[
      Scale[BoundingRectangles[#,threshold,filterSize],
      Dimensions[pyramid[[1]]][[1]]/Dimensions[#][[1]],{0,0}]&,
      pyramid];

OutlineGraphics[grObjects_]:=Graphics[{Opacity[0],Green,EdgeForm[Directive[Green,Thick]],grObjects}]


On[Assert];
Test1=Table[Random[],{i,1,6},{j,1,6}];
Assert[Dimensions[StandardiseImage[Test1//Image,36]][[2]]==36];

(TestB=BoxMatrix[1,5]);
Assert[SobelFilter[TestB,EdgeDir][[2,2]]==0.7853981633974483`]
Assert[SobelFilter[TestB,EdgeDir][[5,5]]==-2.356194490192345`]
Assert[SobelFilter[TestB,EdgeDir][[1,5]]==2.356194490192345`]

Assert[SobelFilter[TestB//N,SurfDir][[2,2]]==2.356194490192345`]
Assert[SobelFilter[TestB//N,SurfDir][[4,4]]==2.356194490192345`]
Assert[SobelFilter[TestB//N,SurfDir][[3,2]]==1.5707963267948966`]
Assert[SobelFilter[TestB//N,SurfDir][[2,3]]==0.`]
Assert[SobelFilter[TestB//N,SurfDir][[4,3]]==0.`]
