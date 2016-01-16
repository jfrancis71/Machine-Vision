(* ::Package:: *)

<<"C:/users/julian/documents/github/Machine-Vision/NeuralNetworks/NeuralNetwork.m"


NNRead["ChessPieces\\ChessPiecesMultiple"]


NNCategoryOutput[probs_,labels_]:=BarChart[probs,ChartLabels->labels];


ChessRecognition[stationNo_Integer]:=
   MobileRecognition[
      (piece=currentImg[[1;;1+64-1;;2,32;;32+64-1;;2]];
      {f=ForwardPropogate[{piece},wl];NNCategoryOutput[f//First,{"Empty","Pawn","Rook","King"}],currentImg//DispImage}
      )&,stationNo
];
