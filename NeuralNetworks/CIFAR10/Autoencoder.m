(* ::Package:: *)

(* Learns a basic denoising autoencoder trained on CIFAR-10
   It doesn't produce the interesting filters reported in the literature.
   However it does seem to do a pretty good job of denoising a corrupted input.
   Loss around 9.9 which presumably corresponds to mean pixel error of around 10%

   Ref: http://deeplearning.net/tutorial/dA.html
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


SeedRandom[1234];

Net1={
   FullyConnected1DTo1D[Table[0.5,{500}],Table[(Random[]-.5)*4*Sqrt[6/(32*32+500)],{500},{32*32}]],
   Logistic,
   FullyConnected1DTo1D[Table[0.,{32*32}],Table[0.,{32*32},{500}]],
   Logistic
};


Images=.2*ColTrainingImages[[All,1]]+.73*ColTrainingImages[[All,2]]+.07*ColTrainingImages[[All,3]];


Fl=Map[Flatten,Images];


TiedGrad[wl_,inputs_,targets_,lossF_]:=(
   t1=ReplacePart[NNGrad[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets,CrossEntropyLoss],{3,2}->wl[[3,2]]*0.0];
   t2=Transpose[NNGrad[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets,CrossEntropyLoss][[3,2]]];
   t3=t1;
   t4=t3;
   t4[[1,2]]+=t2;
   t4)


CrossEntropyLoss[parameters_,inputs_,targets_]:=
   -Total[targets*Log[ForwardPropogation[inputs,parameters]]+(1-targets)*Log[1-ForwardPropogation[inputs,parameters]],2]/Length[inputs]


DeltaLoss[CrossEntropyLoss,outputs_,targets_]:=-((-(1-targets)/(1-outputs)) + (targets/outputs))/Length[outputs];


TiedCrossEntropyLoss[wl_,inputs_,targets_]:=
   CrossEntropyLoss[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets]


TiedRegressionLoss1D[wl_,inputs_,targets_]:=
   RegressionLoss1D[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets]


(* Produced Autoencoder.wdx *)  
Train:=MiniBatchGradientDescent[
      wl,Fl[[1;;10000]],Fl[[1;;10000]],
      NoisyTiedGrad,TiedRegressionLoss1D,
        {MaxEpoch->500000,
         ValidationInputs->Fl[[4001;;4500]],
         ValidationTargets->Fl[[4001;;4500]],         
         InitialLearningRate->.001}];

