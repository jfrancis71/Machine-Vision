(* ::Package:: *)

<<"C:/Users/Julian/Documents/GitHub/Machine-Vision/Potts Model.m"


SegmentationPotential[x_,m_]:=((x-m)^2/(2*0.01))


SegmentationIter[model_,ptch_,mask_]:=
GibbsSampleOnLattice[
model,
Function[{modelI,j,i},
PottsEnergyModel[modelI,j,i]+(1-mask[[j,i]])*SegmentationPotential[ptch[[j,i]],modelI[[3]][[j,i]]/10]]]


Segmentation[ptch_,mask_,iter_:500]:=Nest[SegmentationIter[#,ptch,mask]&,CreatePottsModel[2.3,10,Dimensions[ptch]],iter]


Segmentation[ptch_]:=Segmentation[ptch,ConstantArray[0,ptch//Dimensions]]
