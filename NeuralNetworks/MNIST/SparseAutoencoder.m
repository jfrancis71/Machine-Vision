(* ::Package:: *)

autoencoder={
   FullyConnected1DTo1DInit[28*28,28*28],
   Logistic,
   Sparse[.01,.05],
   FullyConnected1DTo1DInit[28*28,28*28],
   Logistic
};


wl=autoencoder;
TrainingHistory={};
ValidationHistory={};
\[Lambda]=.01;


flat=Map[Flatten,TrainingImages];


Train:=MiniBatchGradientDescent[
   wl,flat[[1;;59000]],flat[[1;;59000]],
   NNGrad,CrossEntropyLoss,
     {MaxEpoch->20000,
      ValidationInputs->flat[[59001;;59500]],
      ValidationTargets->flat[[59001;;59500]],
      StepMonitor->ScreenMonitor,
      InitialLearningRate->\[Lambda]}];
