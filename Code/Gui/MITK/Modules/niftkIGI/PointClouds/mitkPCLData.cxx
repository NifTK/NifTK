/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPCLData.h"


namespace mitk
{


//-----------------------------------------------------------------------------
PCLData::PCLData()
{
}


//-----------------------------------------------------------------------------
PCLData::~PCLData()
{
}


//-----------------------------------------------------------------------------
void PCLData::SetRequestedRegionToLargestPossibleRegion()
{
}


//-----------------------------------------------------------------------------
bool PCLData::RequestedRegionIsOutsideOfTheBufferedRegion()
{
  return false;
}


//-----------------------------------------------------------------------------
bool PCLData::VerifyRequestedRegion()
{
  return true;
}


//-----------------------------------------------------------------------------
void PCLData::SetRequestedRegion(const itk::DataObject* data)
{
}


//-----------------------------------------------------------------------------
void PCLData::SetCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
{
  m_Cloud = cloud;
}


} // namespace
