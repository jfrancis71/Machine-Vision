(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/MVTools.m"


objects=Import["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\DVDImage.jpg"];


{muppets=ImageTake[objects,{114-60,114+80},{90-60,90+60}],aliens=ImageTake[objects,{106-60,106+80},{210-60,210+60}]};


{muppets,aliens}=Map[StandardiseImage,{muppets,aliens}];


DVDRecognition[image_]:=(
   corrMuppets=ImageCorrespondingPoints[muppets//Reverse//Image,image//Reverse//Image];
   corrAlien=ImageCorrespondingPoints[aliens//Image,image//Image];
   If[(Length[corrMuppets[[1]]]+Length[corrAlien[[1]]])>=1,
      If[Length[corrMuppets[[1]]]>Length[corrAlien[[1]]],"Muppets","Alien!"],""])


(*Dynamic[{currentImg//DispImage,out}]*)


(*CameraRecognition[DVDRecognition,256]*)
