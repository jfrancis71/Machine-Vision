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


   int outputOffset, inputOffset;
   int dof, dif;
   for ( int l = 0 ; l < dims[0]; l++ )
      for ( int f = 0 ; f < dims[1] ; f++ )
      {
         outputOffset = OL*l + OF*(f);
         inputOffset = IL*l + IF*(f);

         for ( int y = 0 ; y < dims[2]-2*side ; y++ )
            for ( int x = 0 ; x < dims[3]-2*side ; x++ )
               for ( int sy = 0; sy < filterLength ; sy++ )
               { 
                  dof = outputOffset + OY*(y) + (filterLength*filterLength)*(x) + filterLength*(sy);
                  dif = inputOffset + IY*(y+sy) + x;
                  for ( int sx = 0 ; sx < filterLength ; sx++ )
                     outputs[dof + sx] = inputs[dif + sx];
               }
      }
}

/*
   Input: N*F*(Y+2*SIDE)*(X+2*SIDE) array of doubles
   Output: N*F*Y*X*FLENGTH*FLENGTH array of doubles

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
