/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPCLData.h"
#include <mitkGeometry3D.h>


namespace niftk
{

//-----------------------------------------------------------------------------
PCLData::PCLData()
{
  this->SetGeometry(mitk::Geometry3D::New());
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


//-----------------------------------------------------------------------------
pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr PCLData::GetCloud() const
{
  return m_Cloud;
}


} // namespace
