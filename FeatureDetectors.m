(* ::Package:: *)

(*
Computes log p(Image|H=1)
Implicity assumes sigma=0.05 with an independant
gaussian model per pixel
It is scaled by \[Lambda]
Note this means \[Lambda]=0 will strongly match a blank patch
*)
TemplateDetector[pyr_?PyramidImageQ,kernel_?MatrixQ,mask_?MatrixQ]:=
   Total[mask,2]*Log[1/(Sqrt[2 \[Pi]] .05)] - 1/(2 .05^2) (MVCorrelatePyramid[pyr^2,mask] +
      -2 MVCorrelatePyramid[pyr,kernel*mask] +
      Total[(kernel*mask)^2,2])

TemplateDetector[pyr_?PyramidImageQ,kernel_?MatrixQ,mask_?MatrixQ,\[Lambda]_?PyramidImageQ]:=
   Total[mask,2]*Log[1/(Sqrt[2 \[Pi]] .05)] - 1/(2 .05^2) (MVCorrelatePyramid[pyr^2,mask] +
      -2 \[Lambda] MVCorrelatePyramid[pyr,kernel*mask] +
      \[Lambda]^2 Total[(kernel*mask)^2,2])
