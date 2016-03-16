(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"
<<"C:/users/julian/documents/github/Machine-Vision/FeatureDetectors.m"


topLeftCorner=ArrayPad[ConstantArray[1,{4,4}],{{0,3},{3,0}}];{topLeftCorner//DispImage};


topRightCorner=ArrayPad[ConstantArray[1,{4,4}],{{0,3},{0,3}}];{topRightCorner//DispImage};
bottomLeftCorner=ArrayPad[ConstantArray[1,{4,4}],{{3,0},{3,0}}];{bottomLeftCorner//DispImage};


logistic[z_,s_]:=1/(1+Exp[-(z-s)*10])


edgeKernelHTopLeft=ConstantArray[0,{7,7}];edgeKernelHTopLeft[[4,4;;7]]=1;edgeKernel//DispImage;
edgeKernelVTopLeft=ConstantArray[0,{7,7}];edgeKernelVTopLeft[[1;;4,4]]=1;edgeKernel//DispImage;
edgeKernelHTopRight=ConstantArray[0,{7,7}];edgeKernelHTopRight[[4,1;;4]]=1;edgeKernel//DispImage;
edgeKernelVTopRight=ConstantArray[0,{7,7}];edgeKernelVTopRight[[1;;4,4]]=1;
edgeKernelHBottomLeft=ConstantArray[0,{7,7}];edgeKernelHBottomLeft[[4,4;;7]]=1;edgeKernel//DispImage;
edgeKernelVBottomLeft=ConstantArray[0,{7,7}];edgeKernelVBottomLeft[[4;;7,4]]=1;edgeKernel//DispImage;
edgeKernel//DispImage;


CentralTopLeftKernel=Table[If[Abs[y-5]<=2&&Abs[x+5]<=2,1,0],{y,-8,+8},{x,-8,+8}];
CentralTopRightKernel=Table[If[Abs[y-5]<=2&&Abs[x-5]<=2,1,0],{y,-8,+8},{x,-8,+8}];


ChessboardRecognition[image_]:=(
pyr=BuildPyramid[image];
    topLeft\[Lambda]=MVCorrelatePyramid[pyr,topLeftCorner]/16.;topLeftF//DispImage;
    topRight\[Lambda]=MVCorrelatePyramid[pyr,topRightCorner]/16.;topRightF//DispImage;
bottomLeft\[Lambda]=MVCorrelatePyramid[pyr,bottomLeftCorner]/16.;
    topLeftCons=TemplateDetector[pyr,topLeftCorner,topLeftCorner,topLeft\[Lambda]];
topRightCons=TemplateDetector[pyr,topRightCorner,topRightCorner,topRight\[Lambda]];
bottomLeftCons=TemplateDetector[pyr,bottomLeftCorner,bottomLeftCorner,bottomLeft\[Lambda]];
     edges=SobelFilter[pyr,EdgeMag];

ExtHTopLeft=logistic[MVCorrelatePyramid[logistic[edges,1],edgeKernelHTopLeft],2];
ExtVTopLeft=logistic[MVCorrelatePyramid[logistic[edges,1],edgeKernelVTopLeft],2];
ExtHTopRight=logistic[MVCorrelatePyramid[logistic[edges,1],edgeKernelHTopRight],2];
ExtVTopRight=logistic[MVCorrelatePyramid[logistic[edges,1],edgeKernelVTopRight],2];
ExtHBottomLeft=logistic[MVCorrelatePyramid[logistic[edges,1],edgeKernelHBottomLeft],2];
ExtVBottomLeft=logistic[MVCorrelatePyramid[logistic[edges,1],edgeKernelVBottomLeft],2];
topLeftF=(topLeftCons/33)*(ExtVTopLeft*ExtHTopLeft);
topRightF=(topRightCons/33)*(ExtVTopRight*ExtHTopRight);
bottomLeftF=(bottomLeftCons/33)*(ExtVBottomLeft*ExtHBottomLeft);
     extracttopLeft=Position[topLeftF[[1]],x_/;x>.01];
extracttopRight=Position[topRightF[[1]],x_/;x>.01];
extractbottomLeft=Position[bottomLeftF[[1]],x_/;x>.01];
cent=
logistic[MVCorrelatePyramid[topLeftF,CentralTopLeftKernel],0]*
logistic[MVCorrelatePyramid[topRightF,CentralTopRightKernel],0];
    extractCent=Position[cent[[1]],x_/;x>.5];

)


ChessboardRecognitionOutput[image_]:=(
ChessboardRecognition[image];
Show[image//DispImage,Graphics[{
Red,Map[Point[Reverse[#]]&,extracttopLeft],
Green,Map[Point[Reverse[#]]&,extracttopRight],
Blue,Map[Point[Reverse[#]]&,extractbottomLeft],
Yellow,Map[Point[Reverse[#]]&,extractCent]
}]])
