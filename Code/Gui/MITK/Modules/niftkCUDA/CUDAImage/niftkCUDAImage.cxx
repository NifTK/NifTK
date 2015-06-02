/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCUDAImage.h"

namespace niftk
{

//-----------------------------------------------------------------------------
CUDAImage::CUDAImage()
{
}


//-----------------------------------------------------------------------------
CUDAImage::~CUDAImage()
{
}


//-----------------------------------------------------------------------------
void CUDAImage::SetRequestedRegionToLargestPossibleRegion()
{
}


//-----------------------------------------------------------------------------
bool CUDAImage::RequestedRegionIsOutsideOfTheBufferedRegion()
{
  return false;
}


//-----------------------------------------------------------------------------
bool CUDAImage::VerifyRequestedRegion()
{
  return true;
}


//-----------------------------------------------------------------------------
void CUDAImage::SetRequestedRegion(const itk::DataObject* data)
{
}


//-----------------------------------------------------------------------------
LightweightCUDAImage CUDAImage::GetLightweightCUDAImage() const
{
  return m_LWCImage;
}


//-----------------------------------------------------------------------------
void CUDAImage::SetLightweightCUDAImage(const LightweightCUDAImage& lwci)
{
  m_LWCImage = lwci;
}

} // end namespace
