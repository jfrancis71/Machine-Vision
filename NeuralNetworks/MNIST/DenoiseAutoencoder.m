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


NetDeNoiseTrain:=(
   name="MNIST\\DeNoiseAutoencoder";
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
