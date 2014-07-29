StandardiseImage[image_,size_]:=ImageResize[ColorConvert[image,"GrayScale"],size]


StandardiseImage[image_]:=StandardiseImage[image,128]
StandardiseImage[image_String, size_] := 
 ImageResize[ColorConvert[Import[image], "GrayScale"], size]
