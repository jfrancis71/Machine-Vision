(* ::Package:: *)

(*
   Based on Andrej Karpathy. Not completely faithful implementation. (He uses extra convolution layer and ReLU)
   See: http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html

   On 4K Dataset:
   Iter: 2092
   TrainingLoss: 1.9077
   ValidationLoss: 1.9620
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


SeedRandom[1234];
CIFARNet={
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
   ConvolveFilterBankToFilterBank[Table[
      ConvolveFilterBankTo2D[0,
         Table[Random[],{16},{3},{3}]/(20*5*5)],
      {f,1,16}]],Tanh,
   Adaptor3DTo1D[16,3,3],
   FullyConnected1DTo1D[Table[Random[],{10}],Table[Random[]-.5,{10},{16*3*3}]],
   Softmax
};


CIFAR10NetTrainingInputs=ColTrainingImages[[1;;4000]]*1.;
CIFAR10NetTrainingOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,TrainingLabels[[1;;4000]]];

CIFAR10NetValidationInputs=ColValidationImages[[All]]*1.;
CIFAR10NetValidationOutputs=Map[ReplacePart[ConstantArray[0,10],(#+1)->1]&,ValidationLabels];


wl=CIFARNet;
TrainingHistory={};
ValidationHistory={};


CIFAR10Net1KTanhTrain:=(
   Filename="C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\CIFAR10\\CIFARNet1KTanh.wdx";
   {TrainingHistory,ValidationHistory,wl}=Import[Filename];
   AdaptiveGradientDescent[
      wl,CIFAR10NetTrainingInputs[[1;;1000]],CIFAR10NetTrainingOutputs[[1;;1000]],
      BatchGrad,ClassificationLoss,
        {MaxLoop->500000,
         ValidationInputs->CIFAR10NetValidationInputs,
         ValidationTargets->CIFAR10NetValidationOutputs,
         UpdateFunction->WebMonitor[Filename]}];
)


CIFAR10Net2KTanhTrain:=(
   {TrainingHistory,ValidationHistory,wl}=Import["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\CIFAR10\\CIFARNet2KTanh.wdx"];
   AdaptiveGradientDescent[
      wl,CIFAR10NetTrainingInputs[[1;;2000]],CIFAR10NetTrainingOutputs[[1;;2000]],
      BatchGrad,ClassificationLoss,
        {MaxLoop->500000,
         ValidationInputs->CIFAR10NetValidationInputs,
         ValidationTargets->CIFAR10NetValidationOutputs,
         UpdateFunction->Persist["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\CIFAR10\\CIFARNet2KTanh.wdx"]}];
)


CIFAR10Net4KTanhTrain:=(
   Filename="C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\CIFAR10\\CIFARNet4KTanh.wdx";
   {TrainingHistory,ValidationHistory,wl}=Import[Filename];
   AdaptiveGradientDescent[
      wl,CIFAR10NetTrainingInputs[[1;;4000]],CIFAR10NetTrainingOutputs[[1;;4000]],
      BatchGrad,ClassificationLoss,
        {MaxLoop->500000,
         ValidationInputs->CIFAR10NetValidationInputs,
         ValidationTargets->CIFAR10NetValidationOutputs,
         UpdateFunction->WebMonitor[Filename]}];
)


CIFAR10Output[probs_]:=BarChart[probs[[Reverse[Ordering[probs,-3]]]],ChartLabels->WordLabels[[Reverse[Ordering[probs,-3]]]]];


CIFAR10Outputs[probs_,pics_]:=MapThread[{Image[#2,Interleaving->False],CIFAR10Output[#1]}&,{probs,pics}];
