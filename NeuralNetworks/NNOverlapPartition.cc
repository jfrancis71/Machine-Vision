#include "WolframLibrary.h"

DLLEXPORT mint
WolframLibrary_getVersion()
{
   return WolframLibraryVersion;
}

DLLEXPORT int
WolframLibrary_initialize( WolframLibraryData libData )
{
   return 0;
}

DLLEXPORT void
WolframLibrary_uninitialize( WolframLibraryData libData )
{
   return;
}

DLLEXPORT int
constantzero ( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res )
{
   MArgument_setInteger (Res, 0);
   return LIBRARY_NO_ERROR;
}

void
work( double *inputs, double *outputs, const mint* dims )
{
   int OY = (dims[3]-2)*3*3;
   int OF = (dims[2]-2)*(dims[3]-2)*3*3;
   int OL = dims[1]*(dims[2]-2)*(dims[3]-2)*3*3;
   int IY = dims[3];
   int IF = dims[2]*dims[3];
   int IL = (dims[1]*dims[2]*dims[3]);

   for ( int l = 0 ; l < dims[0]; l++ )
      for ( int f = 0 ; f < dims[1] ; f++ )
         for ( int y = 0 ; y < dims[2]-2 ; y++ )
            for ( int x = 0 ; x < dims[3]-2 ; x++ )
               for ( int sy = 0 ; sy < 3 ; sy++ )
                  for ( int sx = 0 ; sx < 3 ; sx++ )
                     //(*outputs)[l][y][x][sy+1][sx+1] = (*inputs)[l][f][y+sy+1][x+sx+1];
                     outputs[OL*l + OF*(f) + OY*(y) + (3*3)*(x) + 3*(sy) + sx] =
						inputs[IL*l + IF*(f) + IY*(y+sy) + (x+sx)];
}

/*
   Input: N*F*(Y+2)*(X+2) array of doubles
   Output: N*F*Y*X*3*3 array of doubles

   Requires: the data part of the input is Y*X padded on either side by .0 (hence Y+2*X+2)
*/
DLLEXPORT int
NNCPUExtensionOverlappingPartition( WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res )
{
   MTensor inputTensor = MArgument_getMTensor (Args[0]);
   double* inputs = libData -> MTensor_getRealData ( inputTensor );
   mint const* dims;
   dims = libData->MTensor_getDimensions( inputTensor );

   MTensor outputTensor;
   mint type = MType_Real;
   mint dimsOutput[6];
   mint rank = 6;
   int err;
          
   dimsOutput[0] = dims[0];
   dimsOutput[1] = dims[1];
   dimsOutput[2] = dims[2]-2;
   dimsOutput[3] = dims[3]-2;
   dimsOutput[4] = 3;
   dimsOutput[5] = 3;

   err = libData -> MTensor_new ( type, rank, dimsOutput, &outputTensor);
   double* ptr = libData -> MTensor_getRealData ( outputTensor );

   work( inputs, ptr, dims );
  
   MArgument_setMTensor ( Res, outputTensor );
   return LIBRARY_NO_ERROR;
}
