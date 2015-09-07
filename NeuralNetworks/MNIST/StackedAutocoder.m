(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/Autoencoder.m"


<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/MNIST/MNISTData.m"


MNIST=Map[Flatten,TrainingImages]*1.;


StackTrain:=(
   TrainingHistory={};

   layer1Dat=MNIST[[1;;50000]];
   {encoder1,decoder1}=TrainAutoencoder[784,500,layer1Dat,CrossEntropyLoss];

   layer2Dat=ForwardPropogate[layer1Dat,encoder1];
   {encoder2,decoder2}=TrainAutoencoder[500,250,layer2Dat,RegressionLoss1D];

   layer3Dat=ForwardPropogate[layer2Dat,encoder2];
   {encoder3,decoder3}=TrainAutoencoder[250,125,layer3Dat,RegressionLoss1D];

   layer4Dat=ForwardPropogate[layer3Dat,encoder3];
   {encoder4,decoder4}=TrainAutoencoder[125,64,layer4Dat,RegressionLoss1D];

   {encoder,decoder}={Flatten[{encoder1,encoder2,encoder3,encoder4}],Flatten[{decoder1,decoder2,decoder3,decoder4}//Reverse]};
)


SaveStackedAutocoder:=Export["C:\\Users\\Julian\\Google Drive\\Personal\\Computer Science\\WebMonitor\\MNIST\\StackedAutoCoder.wdx",{encoder,decoder}]
