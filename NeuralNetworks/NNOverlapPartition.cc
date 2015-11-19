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
work( double (*inputs)[100][32][34][34], double (*outputs)[100][32][32][32][3][3], const int* dims )
{
   int sizeInputs = sizeof( double )*34*34;
   int sizeOutputs = sizeof( double )*32*32;

   for ( int l = 0 ; l < dims[0]; l++ )
      for ( int f = 0 ; f < 32 ; f++ )
         for ( int y = 0 ; y < 32 ; y++ )
            for ( int x = 0 ; x < 32 ; x++ )
               for ( int sy = -1 ; sy < 2 ; sy++ )
                  for ( int sx = -1 ; sx < 2 ; sx++ )
                     //outputs[l,f,y,x,sy,sx] = inputs[l,f,y+sy,x+sx];
                     (*outputs)[l][y][x][sy+1][sx+1] = (*inputs)[l][f][y+sy+1][x+sx+1];
}

/*
   Input: 100*32*34*34 array of doubles
   Output: 100*32*32*32*3*3 array of doubles

   Requires: the data part of the input is 32*32 padded on either side by .0 (hence 34*34)
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
   dimsOutput[2] = 32;
   dimsOutput[3] = 32;
   dimsOutput[4] = 3;
   dimsOutput[5] = 3;

   err = libData -> MTensor_new ( type, rank, dimsOutput, &outputTensor);
   double* ptr = libData -> MTensor_getRealData ( outputTensor );

   work( inputs, ptr, dims );
  
   MArgument_setMTensor ( Res, outputTensor );
   return LIBRARY_NO_ERROR;
}
