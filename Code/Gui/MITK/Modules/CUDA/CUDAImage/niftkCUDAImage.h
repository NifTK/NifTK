/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCUDAImage_h
#define niftkCUDAImage_h

#include "niftkCUDAExports.h"
#include <CUDAImage/niftkLightweightCUDAImage.h>
#include <mitkBaseData.h>

namespace niftk
{

/**
* A holder for LightweightCUDAImage so that it can be added to a DataNode
* and be made available via DataStorage.
*/
class NIFTKCUDA_EXPORT CUDAImage : public mitk::BaseData
{

public:
  mitkClassMacro(CUDAImage, mitk::BaseData);

  itkFactorylessNewMacro(Self);

  /** @name MITK stuff not applicable. */
  //@{
  /** Does nothing. */
  virtual void SetRequestedRegionToLargestPossibleRegion();
  /** @returns false always */
  virtual bool RequestedRegionIsOutsideOfTheBufferedRegion();
  /** @returns true always */
  virtual bool VerifyRequestedRegion();
  /** Does nothing. */
  virtual void SetRequestedRegion(const itk::DataObject* data);
  //@}

  /**
  * Returns a copy of the LightweightCUDAImage. Remember: LightweightCUDAImage is merely some
  * form of opaque handle.
  * @throws nothing should not throw anything.
  */
  LightweightCUDAImage GetLightweightCUDAImage() const;

  /**
  * Sets the LightweightCUDAImage handle.
  * Remember: LightweightCUDAImage is merely some form of opaque handle.
  * @throws nothing should not throw anything.
  */
  void SetLightweightCUDAImage(const LightweightCUDAImage& lwci);


protected:
  CUDAImage();
  virtual ~CUDAImage();


  /** @name Copy and assignment not allowed. */
  //@{
private:
  CUDAImage(const CUDAImage& copyme);
  CUDAImage& operator=(const CUDAImage& assignme);
  //@}


private:
  LightweightCUDAImage      m_LWCImage;
};

} // end namespace

#endif
