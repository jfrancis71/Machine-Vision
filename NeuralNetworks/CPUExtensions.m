(* ::Package:: *)

Needs["CCompilerDriver`"]


$CCompiler = {"Compiler"->CCompilerDriver`GenericCCompiler`GenericCCompiler, "CompilerInstallation"->"C:\\TDM-GCC-64\\bin","CompilerName"->"gcc.exe"};


src=ReadList["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\NNOverlapPartition.cc",Record,RecordSeparators->{}][[1]];


lib=CreateLibrary[src,"NNOverlapPartition"]


NNOverlapPartition=LibraryFunctionLoad[lib, "NNCPUExtensionOverlappingPartition",{{Real,4}},{Real,6}]


installed=True;


unload:=(LibraryFunctionUnload[NNOverlapPartition];installed=False;)
