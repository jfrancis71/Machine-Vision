(* ::Package:: *)

(*
   Based on Andrej Karpathy. Not completely faithful implementation. (He uses extra convolution layer and ReLU)
   See: http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html

   Needs more training:
   Iter: 155
   TrainingLoss: 2.001
   ValidationLoss: 2.034
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


SeedRandom[1234];
CIFAR10AndrejNet1={
   ConvolveFilterBankToFilterBank[Table[
      ConvolveFilterBankTo2D[0,{
         (Table[Random[],{5},{5}]-.5)/25,
         (Table[Random[],{5},{5}]-.5)/25,
         (Table[Random[],{5},{5}]-.5)/25}/(25*3.)],
      {f,1,20}]],Tanh,
   MaxPoolingFilterBankToFilterBank,
   ConvolveFilterBankToFilterBank[Table[
      ConvolveFilterBankTo2D[0,
         Table[Random[],{20},{5},{5}]/(20*5*5)],
      {f,1,16}]],Tanh,
   MaxPoolingFilterBankToFilterBank,
   Adaptor3DTo1D[16,5,5],
   FullyConnected1DTo1D[Table[Random[],{10}],Table[Random[]-.5,{10},{16*5*5}]],
   Softmax
};


CIFAR10AndrejNet1TrainingInputs=ColTrainingImages[[1;;1000]]*1.;
CIFAR10AndrejNet1TrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;1000]]];

CIFAR10AndrejNet1ValidationInputs=ColValidationImages[[All]]*1.;
CIFAR10AndrejNet1ValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,ValidationLabels];


wl=CIFAR10AndrejNet1;
TrainingHistory={};
ValidationHistory={};


(*{TrainingHistory,ValidationHistory,wl}=Import["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\CIFAR-10\\AndrejNet1.wdx"];*)
(*Export["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\CIFAR-10\\AndrejNet1.wdx",{TrainingHistory,ValidationHistory,wl}];*)


 CIFAR10AndrejNet1Train:=AdaptiveGradientDescent[
   wl,CIFAR10AndrejNet1TrainingInputs,CIFAR10AndrejNet1TrainingOutputs,
   Grad,ClassificationLoss,
     {MaxLoop->500000,
      ValidationInputs->CIFAR10AndrejNet1ValidationInputs,
      ValidationTargets->CIFAR10AndrejNet1ValidationOutputs}];
