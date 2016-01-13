(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


NNRead["ChessPieces\\ChessPiecesMultiple"]


MobileRecognition[
   (piece=currentImg[[1;;1+64-1;;2,32;;32+64-1;;2]];
   {ForwardPropogate[{piece},wl],currentImg//DispImage}
   )&
]



