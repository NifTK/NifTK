/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "CUDAImage.h"


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
  throw std::runtime_error("CUDAImage::SetRequestedRegionToLargestPossibleRegion not supported");
}


//-----------------------------------------------------------------------------
bool CUDAImage::RequestedRegionIsOutsideOfTheBufferedRegion()
{
  throw std::runtime_error("CUDAImage::RequestedRegionIsOutsideOfTheBufferedRegion not supported");
}


//-----------------------------------------------------------------------------
bool CUDAImage::VerifyRequestedRegion()
{
  throw std::runtime_error("CUDAImage::VerifyRequestedRegion not supported");
}


//-----------------------------------------------------------------------------
void CUDAImage::SetRequestedRegion(const itk::DataObject* data)
{
  throw std::runtime_error("CUDAImage::SetRequestedRegion not supported");
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
