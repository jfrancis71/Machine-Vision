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
work( double *inputs, double *outputs, const mint* dims, int side )
{
   int filterLength = side*2+1;
   int OY = (dims[3]-2*side)*filterLength*filterLength;
   int OF = (dims[2]-2*side)*(dims[3]-2*side)*filterLength*filterLength;
   int OL = dims[1]*(dims[2]-2*side)*(dims[3]-2*side)*filterLength*filterLength;
   int IY = dims[3];
   int IF = dims[2]*dims[3];
   int IL = (dims[1]*dims[2]*dims[3]);


   for ( int l = 0 ; l < dims[0]; l++ )
      for ( int f = 0 ; f < dims[1] ; f++ )
         for ( int y = 0 ; y < dims[2]-2*side ; y++ )
            for ( int x = 0 ; x < dims[3]-2*side ; x++ )
               for ( int sy = -side ; sy < 1+side ; sy++ )
                  for ( int sx = -side ; sx < 1+side ; sx++ )
                     //(*outputs)[l][y][x][sy+1][sx+1] = (*inputs)[l][f][y+sy+1][x+sx+1];
                     outputs[OL*l + OF*(f) + OY*(y) + (filterLength*filterLength)*(x) + filterLength*(sy+side) + sx+side] =
						inputs[IL*l + IF*(f) + IY*(y+sy+side) + (x+sx+side)];
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

   mint runLength = MArgument_getInteger(Args[1]);
   int side = (runLength-1)/2; //refers to how far filter extends from centre, e.g. 5*5 filter extends 2 from centre

   dimsOutput[0] = dims[0];
   dimsOutput[1] = dims[1];
   dimsOutput[2] = dims[2]-2*side;
   dimsOutput[3] = dims[3]-2*side;
   dimsOutput[4] = runLength;
   dimsOutput[5] = runLength;

   err = libData -> MTensor_new ( type, rank, dimsOutput, &outputTensor);
   double* ptr = libData -> MTensor_getRealData ( outputTensor );

   work( inputs, ptr, dims, side );
  
   MArgument_setMTensor ( Res, outputTensor );
   return LIBRARY_NO_ERROR;
}
