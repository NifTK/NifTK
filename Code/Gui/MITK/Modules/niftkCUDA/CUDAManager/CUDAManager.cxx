/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "CUDAManager.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


//-----------------------------------------------------------------------------
static bool CUDADelayLoadCheck()
{
  // the cuda dlls are delay-loaded, that means they are only mapped into our process
  // on demand, i.e. when we try to call an exported function for the first time.
  // this is what we do here.
  // if it fails (e.g. dll not found) then the runtime linker will throw a SEH exception.
  __try
  {
    // touch an entry point in nvcuda.dll
    int   driverversion = 0;
    CUresult r = cuDriverGetVersion(&driverversion);
    // touch an entry point in cudart*.dll
    int   runtimeversion = 0;
    cudaError_t s = cudaRuntimeGetVersion(&runtimeversion);

    // we dont care about the result.
    // (actually, it would make sense to check that the driver supports cuda 5).

    return true;
  }
  __except(1)
  {
    return false;
  }

  // unreachable
  assert(false);
}


//-----------------------------------------------------------------------------
CUDAManager*      CUDAManager::s_Instance = 0;


//-----------------------------------------------------------------------------
CUDAManager* CUDAManager::GetInstance()
{
  if (s_Instance == 0)
  {
    bool  ok = CUDADelayLoadCheck();
    if (!ok)
    {
      throw std::runtime_error("No CUDA available. Delay-load check failed.");
    }

    s_Instance = new CUDAManager;
  }


  return s_Instance;
}


//-----------------------------------------------------------------------------
CUDAManager::CUDAManager()
{

}


//-----------------------------------------------------------------------------
CUDAManager::~CUDAManager()
{
}
