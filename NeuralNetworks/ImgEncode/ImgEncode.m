(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/CIFAR10/CIFAR10Data.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


Images=Sqrt[Map[Reverse[Apply[Plus,#^2]]&,ColTrainingImages[[1;;4000]]]]/3.;


SparseRandom[]:=If[Random[]>.995,Random[],Random[]/1000]


MNIST=Map[Flatten,TrainingImages]*1.;


SeedRandom[1234];
Net1={
   FullyConnected1DTo1D[Table[Random[],{100}],Table[Random[]-.5,{100},{32*32}]],
   Tanh,
   FullyConnected1DTo1D[Table[Random[],{32*32}],Table[Random[]-.5,{32*32},{100}]]
};

Net2={
   FullyConnected1DTo1D[Table[-10,{100}]-.5,Table[(Random[]-.5),{100},{32*32}]/(1000*32*32)],
   Logistic,
   FullyConnected1DTo1D[Table[.2759,{32*32}],Table[Random[]-.5,{32*32},{100}]/1000000]
};

Net3={
   FullyConnected1DTo1D[Table[Random[]-.5,{100}]-.5,Table[(Random[]-.5),{100},{32*32}]/(1000*32*32)],
   FullyConnected1DTo1D[Table[Random[]-.5,{32*32}],Table[Random[]-.5,{32*32},{100}]/1000000]
};

Net4={
   FullyConnected1DTo1D[Table[0.5,{100}],Table[SparseRandom[],{100},{32*32}]/(1000*32*32)],
   Logistic,
   FullyConnected1DTo1D[Table[Random[]-.5,{32*32}],Table[Random[]-.5,{32*32},{100}]/1000000]
};

Net5={
   FullyConnected1DTo1D[Table[0.5,{100}],Table[ReplacePart[ConstantArray[0.,{32*32}],1+RandomInteger[(32*32)-1]->1.],{100}]],
   Logistic,
   FullyConnected1DTo1D[Table[Random[]-.5,{32*32}],Table[Random[]-.5,{32*32},{100}]/1000000]
};

Net6={
   FullyConnected1DTo1D[Table[0.,{100}],Table[(Random[]-.5)*Sqrt[2/(32*32)],{100},{32*32}]],
   Tanh,
   FullyConnected1DTo1D[Table[0.,{32*32}],Table[(Random[]-.5)*Sqrt[2/100],{32*32},{100}]]
};

Net7={
   FullyConnected1DTo1D[Table[0.,{100}],Table[(Random[]-.5)*Sqrt[2/(32*32)],{100},{32*32}]],
   Tanh,
   FullyConnected1DTo1D[Table[0.,{32*32}],Table[0.,{32*32},{100}]]
};

Net8={
   FullyConnected1DTo1D[Table[0.5,{500}],Table[(Random[]-.5)*4*Sqrt[6/(28*28+500)],{500},{28*28}]],
   Logistic,
   FullyConnected1DTo1D[Table[0.,{28*28}],Table[0.,{28*28},{500}]],
   Logistic
};

Net9={
   FullyConnected1DTo1D[Table[0.,{500}],Table[(Random[]-.5)*4*Sqrt[6/(32*32+500)],{500},{32*32}]],
   FullyConnected1DTo1D[Table[0.,{32*32}],Table[0.,{32*32},{500}]],
};



wl=Net2;
TrainingHistory={};
ValidationHistory={};


Fl=Map[Flatten,Images];


AddNoise[inputs_]:=(
   tmp1=UnitStep[RandomReal[{0,1},inputs//Dimensions]-.667];
   (inputs*(1-tmp1)+tmp1*RandomReal[{0,1},inputs//Dimensions])
)


Net1KTanhTrain:=(
   name="ImgEncode\\Net1Sz1KTanh";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->SkipWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net2Sz1KLogTrain:=(
   name="ImgEncode\\Net2Sz1KLog";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->50000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net3Sz1KTrain:=(
   name="ImgEncode\\Net3Sz1K";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->50000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net4Sz1KTrain:=(
   name="ImgEncode\\Net4Sz1K";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


NetDenoise1Train:=(
   name="ImgEncode\\NetDenoise1";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]]+noise,Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net5Train:=(
   name="ImgEncode\\Net5";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]]+noise,Fl[[1;;1000]],
      BatchGrad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)

Net6Train:=(
   name="ImgEncode\\Net6";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   GradientDescent[
      wl,Fl[[1;;100]]+noise[[1;;100]],Fl[[1;;100]],
      Grad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,500],
         InitialLearningRate->\[Lambda]}];
)


Net6Train:=(
   name="ImgEncode\\Net6";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   MBGDWithNoise[
      wl,Fl[[1;;4000]],Fl[[1;;4000]],
      Grad,RegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


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


Net7GradTrain:=(
   name="ImgEncode\\Net7Grad";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   GradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      TiedGrad,TiedRegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net7AdaptTrain:=(
   name="ImgEncode\\Net7AdaptGrad";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   AdaptiveGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      TiedGrad,TiedRegressionLoss1D,
        {MaxLoop->500000,
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


NoisyTiedGrad[wl_,inputs_,targets_,lossF_]:=
   TiedGrad[wl,AddNoise[inputs],targets,lossF];


Net7NoiseTrain:=(
   name="ImgEncode\\Net7Noise";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   MiniBatchGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      NoisyTiedGrad,TiedRegressionLoss1D,
        {MaxLoop->500000,
         ValidationInputs->Fl[[3000;;3200]],
         ValidationTargets->Fl[[3000;;3200]],         
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)


Net8Train:=(
   name="ImgEncode\\Net8";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   MiniBatchGradientDescent[
      wl,MNIST[[1;;50000]],MNIST[[1;;50000]],
      TiedGrad,TiedCrossEntropyLoss,
        {MaxLoop->500000,
         ValidationInputs->MNIST[[53000;;53200]],
         ValidationTargets->MNIST[[53000;;53200]],         
         UpdateFunction->WebMonitor[name],
         InitialLearningRate->\[Lambda]}];
)



Net8NoiseTrain:=(
   name="ImgEncode\\Net8Noise";
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

Net9Train:=(
   name="ImgEncode\\Net9";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   MiniBatchGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      TiedGrad,TiedCrossEntropyLoss,
        {MaxLoop->500000,
         ValidationInputs->Fl[[3000;;3200]],
         ValidationTargets->Fl[[3000;;3200]],         
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)

Net9NoiseTrain:=(
   name="ImgEncode\\Net9Noise";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   MiniBatchGradientDescent[
      wl,Fl[[1;;1000]],Fl[[1;;1000]],
      TiedGrad,TiedCrossEntropyLoss,
        {MaxLoop->500000,
         ValidationInputs->Fl[[3000;;3200]],
         ValidationTargets->Fl[[3000;;3200]],         
         UpdateFunction->CheckpointWebMonitor[name,100],
         InitialLearningRate->\[Lambda]}];
)
