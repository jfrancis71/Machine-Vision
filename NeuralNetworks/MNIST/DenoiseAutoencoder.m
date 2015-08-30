(* ::Package:: *)

(*
   Denoising autoencoder produces similar filters to that reported in the literature
   Cross entropy loss around 33 and 35, training and validation respectively after 52 epochs.
   Qualitatively seems to do a good job of denoising reconstruction.

   Implementaton based on Ref: http://deeplearning.net/tutorial/dA.html   
*)


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/Autoencoder.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNIST=Map[Flatten,TrainingImages]*1.;


SeedRandom[1234];

Net1={
   FullyConnected1DTo1D[Table[0.5,{500}],Table[(Random[]-.5)*4*Sqrt[6/(28*28+500)],{500},{28*28}]],
   Logistic,
   FullyConnected1DTo1D[Table[0.,{28*28}],Table[0.,{28*28},{500}]],
   Logistic
};


TiedGrad[wl_,inputs_,targets_,lossF_]:=(
   t1=ReplacePart[NNGrad[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets,CrossEntropyLoss],{3,2}->wl[[3,2]]*0.0];
   t2=Transpose[NNGrad[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets,CrossEntropyLoss][[3,2]]];
   t3=t1;
   t4=t3;
   t4[[1,2]]+=t2;
   t4)


TiedRegressionLoss1D[wl_,inputs_,targets_]:=
   RegressionLoss1D[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets]


CrossEntropyLoss[parameters_,inputs_,targets_]:=
   -Total[targets*Log[ForwardPropogation[inputs,parameters]]+(1-targets)*Log[1-ForwardPropogation[inputs,parameters]],2]/Length[inputs]


DeltaLoss[CrossEntropyLoss,outputs_,targets_]:=-((-(1-targets)/(1-outputs)) + (targets/outputs))/Length[outputs];


TiedCrossEntropyLoss[wl_,inputs_,targets_]:=
   CrossEntropyLoss[ReplacePart[wl,{3,2}->Transpose[wl[[1,2]]]],inputs,targets]


NetDeNoiseTrain:=(
   name="ImgEncode\\DeNoiseAutoencoder";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   MiniBatchGradientDescent[
      wl,MNIST[[1;;50000]],MNIST[[1;;50000]],
      NoisyTiedGrad,TiedCrossEntropyLoss,
        {MaxEpoch->500000,
         ValidationInputs->MNIST[[53000;;53200]],
         ValidationTargets->MNIST[[53000;;53200]],         
         UpdateFunction->CheckpointWebMonitor[name,5],
         InitialLearningRate->\[Lambda]}];
)
