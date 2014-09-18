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
#include <CUDAImage/LightweightCUDAImage.h>
#include <mitkBaseData.h>


// BaseData is rather fat. can we avoid it?
class NIFTKCUDA_EXPORT CUDAImage : public mitk::BaseData
{

public:
  mitkClassMacro(CUDAImage, mitk::BaseData);

  itkFactorylessNewMacro(Self);

  // mitk stuff not applicable. will always throw an exception, or fail somehow.
  virtual void SetRequestedRegionToLargestPossibleRegion();
  virtual bool RequestedRegionIsOutsideOfTheBufferedRegion();
  virtual bool VerifyRequestedRegion();
  virtual void SetRequestedRegion(const itk::DataObject* data);


  LightweightCUDAImage GetLightweightCUDAImage() const;
  void SetLightweightCUDAImage(const LightweightCUDAImage& lwci);


protected:
  CUDAImage();
  virtual ~CUDAImage();


private:
  CUDAImage(const CUDAImage& copyme);
  CUDAImage& operator=(const CUDAImage& assignme);


private:
  LightweightCUDAImage      m_LWCImage;
};


#endif // CUDAImage_h
