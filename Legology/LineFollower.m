(* ::Package:: *)

(* ::Subtitle:: *)
(*Line Follower*)


(* ::Text:: *)
(*Simple Line Following Program. Haven't checked it recently since some small changes*)
(*but basic idea is to boot up library and attach to EV3.*)
(*EV3 will follow a narrow dark line around a course. Don't forget to set camera address.*)
(*Kick of by calling LineFollower.*)
(*Note, EV3 doesn't track well when there are sudden changes in line direction. Problem*)
(*is line disappears from camera view.*)
(**)
(*Procedure:*)
(*Set camera variable*)
(*Bind to port*)
(*Optional: Setup a pushbutton to control the driveStatus variable *)
(*Call LineFollower[]*)


(* ::Input:: *)
(*camera="http://192.168.1.184/image.jpg";*)


(* ::Input:: *)
(*SetDirectory["c:/users/julian/secure/legotastic/"]*)


(* ::Input:: *)
(*findCentre[scanLine_]:=*)
(*Ordering[ListConvolve[{1,1,1,1,1}/5,scanLine]]//First*)
(**)


(* ::Input:: *)
(*<<EV3Library`*)
(**)
(*LineFollower[]:=( *)
(*EV3StartMotor[EV3PortB];*)
(*EV3StartMotor[EV3PortC];*)
(*driveStatus="start";*)
(*blackbox={};*)
(*While[driveStatus=="start",img=ColorConvert[Import[camera],"GrayScale"];*)
(*scanLine = ImageData[img,DataReversed->True][[1]];*)
(*centre=findCentre[scanLine];*)
(*(*Note left means turn left *)*)
(*If[centre > 170,right=True;left=False,*)
(*If[centre<130,right=False;left=True,left=False;right=False]];*)
(*If[left,EV3StopMotor[EV3PortB],EV3StartMotor[EV3PortB]];*)
(*If[right,EV3StopMotor[EV3PortC],EV3StartMotor[EV3PortC]];*)
(*AppendTo[blackbox,{img,scanLine,centre,left,right}];*)
(*Pause[0.1];];*)
(*StopDrive[])*)
(**)
(*StopDrive[]:=(EV3StopMotor[EV3PortB];*)
(*EV3StopMotor[EV3PortC];)*)
