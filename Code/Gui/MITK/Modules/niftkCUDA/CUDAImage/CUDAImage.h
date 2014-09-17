/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef CUDAImage_h
#define CUDAImage_h

#include "niftkCUDAExports.h"
#include <mitkBaseData.h>
#include <cuda_runtime_api.h>


// BaseData is rather fat. can we avoid it?
class NIFTKCUDA_EXPORT CUDAImage : public mitk::BaseData
{

public:
  mitkClassMacro(CUDAImage, mitk::BaseData)



protected:
  CUDAImage();
  virtual ~CUDAImage();


private:
  CUDAImage(const CUDAImage& copyme);
  CUDAImage& operator=(const CUDAImage& assignme);


private:
  // remember: CUevent is interchangable with cudaEvent_t and vice versa

  int             m_Device;    // the device this image lives on.
  unsigned int    m_Id;
  cudaEvent_t     m_ReadyEvent;       // signaled when the image is ready for consumption.

  unsigned int    m_Width;
  unsigned int    m_Height;
  unsigned int    m_BytePitch;
  // FIXME: pixel type descriptor
};


#endif // CUDAImage_h
