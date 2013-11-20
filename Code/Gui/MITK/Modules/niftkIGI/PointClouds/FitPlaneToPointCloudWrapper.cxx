/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "FitPlaneToPointCloudWrapper.h"
#include <mitkPointSetReader.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>


namespace niftk
{


//-----------------------------------------------------------------------------
FitPlaneToPointCloudWrapper::FitPlaneToPointCloudWrapper()
  : m_PlaneCoefficients(new pcl::ModelCoefficients)
  , m_MinPlaneDistance(std::numeric_limits<float>::max())
  , m_MaxPlaneDistance(-std::numeric_limits<float>::max())
  , m_AvgPlaneDistance(0)
  , m_RmsPlaneDistance(0)
{
}


//-----------------------------------------------------------------------------
FitPlaneToPointCloudWrapper::~FitPlaneToPointCloudWrapper()
{
}


//-----------------------------------------------------------------------------
void FitPlaneToPointCloudWrapper::OutputParameters(std::ostream& log)
{
  if (m_PlaneCoefficients->values.size() != 4)
  {
    throw std::logic_error("Need to call FitPlane() first!");
  }

  log << "# plane coefficients:" << std::endl;
  log << m_PlaneCoefficients->values[0] << ' ' << m_PlaneCoefficients->values[1] << ' ' << m_PlaneCoefficients->values[2] << ' ' << m_PlaneCoefficients->values[3] << std::endl;

  log << "# minimum distance to estimated plane: " << m_MinPlaneDistance << std::endl
      << "# maximum distance to estimated plane: " << m_MaxPlaneDistance << std::endl
      << "# average distance to estimated plane: " << m_AvgPlaneDistance << std::endl
      << "# rms distance to estimated plane: " << m_RmsPlaneDistance << std::endl
  ;
}


//-----------------------------------------------------------------------------
void FitPlaneToPointCloudWrapper::FitPlane(const std::string& filename)
{
  if (filename.empty())
    throw std::runtime_error("Point cloud file name cannot be empty");

  // read .mps file with mitk's build-in mechanism (very slow).
  mitk::PointSetReader::Pointer   psreader = mitk::PointSetReader::New();
  psreader->SetFileName(filename);
  psreader->Update();
  mitk::PointSet::Pointer pointset = psreader->GetOutput();
  if (pointset.IsNull())
    throw std::runtime_error("Could not read point set file " + filename);

  // now convert it to a pcl representation.
  // this is infact a simple std::vector with all the points.
  pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (mitk::PointSet::PointsConstIterator i = pointset->Begin(); i != pointset->End(); ++i)
  {
    const mitk::PointSet::PointType& p = i->Value();
    cloud->push_back(pcl::PointXYZ(p[0], p[1], p[2]));
  }

  // this is effectively the same as in the pcl tutorials.
  pcl::SACSegmentation<pcl::PointXYZ>   seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);         // arbitrary?
  seg.setInputCloud(cloud);

  pcl::PointIndices::Ptr        inliers(new pcl::PointIndices);
  seg.segment(*inliers, *m_PlaneCoefficients);

  if (inliers->indices.size () == 0)
  {
    throw std::runtime_error("Could not estimate a planar model for the given dataset.");
  }

  // sanity check
  if (m_PlaneCoefficients->values.size() != 4)
  {
    throw std::runtime_error("Plane estimation did not come out with the expected 4 parameters");
  }

  // compute distance of each point to the fitted plane.
  float   m_MinPlaneDistance =  std::numeric_limits<float>::max();
  float   m_MaxPlaneDistance = -std::numeric_limits<float>::max();
  float   m_AvgPlaneDistance = 0;
  float   m_RmsPlaneDistance = 0;
  float   planecoeffthingy = std::sqrt(m_PlaneCoefficients->values[0] * m_PlaneCoefficients->values[0] +  m_PlaneCoefficients->values[1] * m_PlaneCoefficients->values[1] + m_PlaneCoefficients->values[2] * m_PlaneCoefficients->values[2]);
  for (pcl::PointCloud<pcl::PointXYZ>::const_iterator i = cloud->begin(); i != cloud->end(); ++i)
  {
    float   dist = std::abs(m_PlaneCoefficients->values[0] * i->x + m_PlaneCoefficients->values[1] * i->y + m_PlaneCoefficients->values[2] * i->z + m_PlaneCoefficients->values[3]);
    dist /= planecoeffthingy;

    m_RmsPlaneDistance += dist * dist;
    m_AvgPlaneDistance += dist;
    m_MinPlaneDistance = std::min(m_MinPlaneDistance, dist);
    m_MaxPlaneDistance = std::max(m_MaxPlaneDistance, dist);
  }
  m_AvgPlaneDistance /= cloud->size();
  m_RmsPlaneDistance = std::sqrt(m_RmsPlaneDistance / cloud->size());

  // done.
}


} // namespace
