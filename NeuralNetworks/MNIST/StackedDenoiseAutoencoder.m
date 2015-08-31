(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/Autoencoder.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNIST=Map[Flatten,TrainingImages]*1.;


ReadNN["MNIST","DenoiseAutoencoder"];Net1=wl;
ReadNN["MNIST","StackedDenoiseAutoencoder"];


layer1Dat=ForwardPropogation[MNIST,Net1[[1;;2]]];


SeedRandom[1234];

Net2={
   FullyConnected1DTo1DInit[500,250],
   Logistic,
   FullyConnected1DTo1DInit[250,500],
   Logistic
};


StackedDeNoiseTrain:=(
   name="MNIST\\StackedDeNoiseAutoencoder";
   {TrainingHistory,ValidationHistory,wl,\[Lambda]}=Import[StringJoin["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\",name,".wdx"]];
   MiniBatchGradientDescent[
      wl,layer1Dat[[1;;50000]],layer1Dat[[1;;50000]],
      NoisyTiedGrad,TiedRegressionLoss1D,
        {MaxEpoch->500000,
         ValidationInputs->layer1Dat[[53000;;53200]],
         ValidationTargets->layer1Dat[[53000;;53200]],         
         UpdateFunction->CheckpointWebMonitor[name,5],
         InitialLearningRate->\[Lambda]}];
)


NNStack={
   Net1[[1]],
   Logistic,
   wl[[1]],
   Logistic
};


NNDeStack={
   FullyConnected1DTo1D[wl[[3,1]],Transpose[wl[[1,2]]]],
   Logistic,
   FullyConnected1DTo1D[Net1[[3,1]],Transpose[Net1[[1,2]]]],
   Logistic
};

