(* ::Package:: *)

Needs["CCompilerDriver`"]


$CCompiler = {"Compiler"->CCompilerDriver`GenericCCompiler`GenericCCompiler, "CompilerInstallation"->"C:\\TDM-GCC-64\\bin","CompilerName"->"gcc.exe"};


src=ReadList["C:\\Users\\Julian\\Documents\\GitHub\\Machine-Vision\\NeuralNetworks\\NNOverlapPartition.cc",Record,RecordSeparators->{}][[1]];
lib=CreateLibrary[src,"NNOverlapPartition","Debug"->True]


NNOverlapPartition=LibraryFunctionLoad[lib, "NNCPUExtensionOverlappingPartition",{{Real,4},Integer},{Real,6}];


NNMaxListable=LibraryFunctionLoad[lib, "NNCPUExtensionMaxListable",{{Real,6}},{Real,4}];


installed=True;


unload:=(
   LibraryFunctionUnload[NNOverlapPartition];
   LibraryFunctionUnload[NNMaxListable];
   installed=False;)


unload
