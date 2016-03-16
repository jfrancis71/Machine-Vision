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


NeighbourProb[x1_,x2_]=0.3*(15.9/(1.+(2500.*(x1-x2)^2)))+0.7;


UniformProdGaussS[S1_,S2_,S3_,countP_,lowerLimit_,upperLimit_]:=
Log[Clip[-1/(Sqrt[2 \[Pi]] .05)^countP .886 * Exp[-S1 + S2^2/S3] ( Erf[Clip[( lowerLimit S3-S2)/Sqrt[S3],{-5,5}]] + Erf[Clip[(S2-upperLimit S3)/Sqrt[S3],{-5,5}]] )/(2 Sqrt[S3]),{0.00001,\[Infinity]}]]



(* Computes a probability distribution over pixels in patch specified by mask. Assumes relative brightness as specified by kernel, where the lower limit is denoted by lowerLimit.
The model assumes a uniform distribution of brightness from lowerLimit to 1, and each pixel is modelled as gaussian distributed centered on kernel*brightness with a standard deviation of .05.
  
Note, for many purposes this is not a particularly convenient model, particularly
for further stages in marginalisation.
Also note, it can give 0 as a probability if the value lies outside the range. This can be undesirable for many applications
 *)


UniformProdGauss[pyr_?PyramidImageQ,kernel_,mask_,l_,u_]:=(
countP = Total[mask,2];
    NS1 = (1./(2. .0025)) MVCorrelatePyramid[pyr^2,mask*kernel];
    NS2 = (1./(2. .0025))  MVCorrelatePyramid[pyr,mask*kernel];
    NS3 = (1./(2. .0025)) countP;
UniformProdGaussS[NS1,NS2,NS3,countP,l,u])
