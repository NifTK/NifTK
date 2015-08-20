/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkAddTwoIntegers.h"
#include <niftkCUDAUtils.h>

namespace niftk
{

//-----------------------------------------------------------------------------
__global__ void add_two_integers(int *a, int *b, int *c)
{
  *c = *a + *b;
}


//-----------------------------------------------------------------------------
int AddTwoIntegers(int a, int b)
{
  int c;
  int *dev_a, *dev_b, *dev_c;
  int size = sizeof( int );

  niftkCUDACall( cudaMalloc( (void**)&dev_a, size ));
  niftkCUDACall( cudaMalloc( (void**)&dev_b, size ));
  niftkCUDACall( cudaMalloc( (void**)&dev_c, size ));

  niftkCUDACall( cudaMemcpy( dev_a, &a, size, cudaMemcpyHostToDevice ));
  niftkCUDACall( cudaMemcpy( dev_b, &b, size, cudaMemcpyHostToDevice ));

  add_two_integers<<< 1, 1 >>>( dev_a, dev_b, dev_c );

  niftkCUDACall( cudaMemcpy( &c, dev_c, size, cudaMemcpyDeviceToHost ));

  niftkCUDACall( cudaFree( dev_a ));
  niftkCUDACall( cudaFree( dev_b ));
  niftkCUDACall( cudaFree( dev_c ));

  return c;
}

} // end namespace
