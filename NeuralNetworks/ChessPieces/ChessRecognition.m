(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


NNRead["ChessPieces\\ChessPiecesMultiple"]


ChessRecognition[stationNo_Integer]:=
   MobileRecognition[
      (piece=currentImg[[1;;1+64-1;;2,32;;32+64-1;;2]];
      {ForwardPropogate[{piece},wl],currentImg//DispImage}
      )&,stationNo
];
