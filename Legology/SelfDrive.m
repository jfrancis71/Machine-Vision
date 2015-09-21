(* ::Package:: *)

(* ::Subtitle:: *)
(*Self Drive*)


(* ::Text:: *)
(*Program will drive EV3 about on the floor attempting to avoid obstacles*)
(*On startup it looks at a scanline in front of it and takes this as its reference.*)
(*As it drives on if it encounters any scan lines significantly different (after smoothing) it will rotate.*)
(**)
(*Procedure:*)
(*Call MainGUI[]*)
(*Call SelfDrive[]*)
(**)


(* ::Input:: *)
(*(**)
(*Timings 16^^10*)
(*Straight 3.13 cm/sec*)
(*Turn 9.33 degrees/sec*)
(*This was for the tracker robot with speed 16^^10 (I think!)*)
(**)*)


MainGUI[]:=
Dynamic[{
Show[img,Graphics[{Green,Line[{{0,300},{360,300}}]}],ImageSize->Small],
ListPlot[current,PlotRange->{0,1}],NumberForm[left,{3,2}],NumberForm[right,{3,2}],
Button["STOP",driveStatus="stop"]
}]


<<EV3Library`

SelfDrive[] := (
  seq = {};
  surface = 
   ImageData[
     ColorConvert[Import["http://172.31.24.139/image.jpg"], 
      "GrayScale"],DataReversed->True][[300]];
  EV3StartMotor[EV3PortB];
  EV3StartMotor[EV3PortC];
  driveStatus = "start";
  While[driveStatus=="start",
   img = ColorConvert[Import["http://172.31.24.139/image.jpg"], 
     "GrayScale"];
   seqSurface = {};
   AppendTo[seq, img];
   current = ImageData[img,DataReversed->True][[300]];
   AppendTo[seqSurface, current];
   diffSurface = Abs[current - surface];
   smoothed = ListConvolve[{1, 1, 1, 1, 1}/5, diffSurface];
   left = Max[smoothed[[1 ;; 180]]];
   right = Max[smoothed[[181 ;; 356]]];
   If[left>0.2 && right>0.2,left=0.0];
   If[left > 0.2, EV3StopMotor[EV3PortC], EV3StartMotor[EV3PortC]];
   If[right > 0.2, EV3StopMotor[EV3PortB], EV3StartMotor[EV3PortB]];
   If[left>0.2||right>0.2,Pause[0.5],Pause[0.2]];
   ];
  StopDrive[]
)

StopDrive[] := (
  EV3StopMotor[EV3PortB];
  EV3StopMotor[EV3PortC];
  )
