/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef CUDAManager_h
#define CUDAManager_h

#include "niftkCUDAExports.h"
#include <CUDAImage/CUDAImage.h>
#include <QThread>
#include <cuda.h>



struct ScopedCUDADevice
{
  ScopedCUDADevice(int dev);
  ~ScopedCUDADevice();
};


class NIFTKCUDA_EXPORT CUDAManager : public QThread
{

public:
  /**
   *
   * @throws std::runtime_error if CUDA is not available on the system.
   */
  static CUDAManager* GetInstance();

  ScopedCUDADevice ActivateDevice(int dev);


protected:
  CUDAManager();
  virtual ~CUDAManager();


private:
  static CUDAManager*      s_Instance;
};


#endif // CUDAManager_h
