(* ::Package:: *)

actorCont=Import["C:\\Users\\Julian\\ImageDataSetsPublic\\NFaceScrub\\facescrub_actors.txt"];


actorDat=ReadList[StringToStream[actorCont],String,RecordSeparators->"\n"];Length[actorDat]


actressCont=Import["C:\\Users\\Julian\\ImageDataSetsPublic\\NFaceScrub\\facescrub_actresses.txt"];


actressDat=ReadList[StringToStream[actressCont],String,RecordSeparators->"\n"];Length[actressDat]


BoundingBox[recordNo_Integer,database_]:=( 
   c=ReadList[StringToStream[database[[recordNo]]],Record,RecordSeparators->"\t"][[5]];
   Map[FromDigits,ReadList[StringToStream[c],Record,RecordSeparators->","]])


ReadURL[recordNo_Integer,database_]:=(
   fileName=ReadList[StringToStream[database[[recordNo]]],Record,RecordSeparators->"\t"][[4]];
   Import[fileName]
)


FaceExtract[recordNo_Integer]:=(o=ReadRecord[recordNo];hght=ImageDimensions[o[[2]]][[2]];ImageTrim[o[[2]],{{o[[1,1]],N[hght]-o[[1,2]]},{o[[1,3]],N[hght]-o[[1,4]]}}])


dump[database_,from_,to_,dir_]:=Table[
   r=ReadURL[t,database];
   If[Not[r===$Failed],
      Export[dir<>"\\"<>ToString[t]<>".jpg",r]],
   {t,from,to}]


names=FileNames["C:\\Users\\Julian\\ImageDataSetsPublic\\faceScrub\\images\\*.jpg"];


NNImageTrim[image_,{{x1_,y1_},{x2_,y2_}}]:=( 
   hght=ImageDimensions[image][[2]];
   ImageTrim[image,{{x1,hght-y1},{x2,hght-y2}}])


FaceSample[database_,from_,to_,sourcedir_,destdir_]:=Table[
   (bb=BoundingBox[t,database];
   size=bb[[3]]-bb[[1]];
   centx=(bb[[1]]+bb[[3]])/2;
   centy=(bb[[2]]+bb[[4]])/2;
   {centx,centy}={centx,centy}+({Random[],Random[]}-{.5,.5})*size*.25;
   size=size*(1+(Random[]-.5)*.5);
   img=Import[sourcedir<>"\\"<>ToString[t]<>".jpg"];
   If[Not[img===$Failed],
      Export[
         destdir<>"\\"<>ToString[t]<>".jpg",
         ColorConvert[w[t]=ImageResize[NNImageTrim[img,
         {{centx-(size/2),centy-(size/2)},{centx+(size/2),centy+(size/2)}}
         ],{32,32}],"GrayScale"]
   ]];)
   ,{t,from,to}]
