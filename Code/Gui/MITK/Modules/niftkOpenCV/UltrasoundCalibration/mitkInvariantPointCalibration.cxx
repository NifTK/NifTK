/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkInvariantPointCalibration.h"
#include <mitkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <mitkCameraCalibrationFacade.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkExceptionMacro.h>
#include <iostream>

namespace mitk {

//-----------------------------------------------------------------------------
InvariantPointCalibration::InvariantPointCalibration()
: m_CostFunction(NULL)
, m_PointData(NULL)
, m_TrackingData(NULL)
{
  m_RigidTransformation.resize(6);
  m_RigidTransformation[0] = 0;
  m_RigidTransformation[1] = 0;
  m_RigidTransformation[2] = 0;
  m_RigidTransformation[3] = 0;
  m_RigidTransformation[4] = 0;
  m_RigidTransformation[5] = 0;
}


//-----------------------------------------------------------------------------
InvariantPointCalibration::~InvariantPointCalibration()
{
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetInvariantPoint(const mitk::Point3D& point)
{
  m_CostFunction->SetInvariantPoint(point);
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::Point3D InvariantPointCalibration::GetInvariantPoint() const
{
  return m_CostFunction->GetInvariantPoint();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetOptimiseInvariantPoint(const bool& optimise)
{
  m_CostFunction->SetOptimiseInvariantPoint(optimise);
  this->Modified();
}


//-----------------------------------------------------------------------------
bool InvariantPointCalibration::GetOptimiseInvariantPoint() const
{
  return m_CostFunction->GetOptimiseInvariantPoint();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetTimingLag(const TimeStampType& timeStamp)
{
  m_CostFunction->SetTimingLag(timeStamp);
  this->Modified();
}


//-----------------------------------------------------------------------------
InvariantPointCalibration::TimeStampType InvariantPointCalibration::GetTimingLag()
{
  return m_CostFunction->GetTimingLag();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetOptimiseTimingLag(const bool& optimise)
{
  m_CostFunction->SetOptimiseTimingLag(optimise);
  this->Modified();
}


//-----------------------------------------------------------------------------
bool InvariantPointCalibration::GetOptimiseTimingLag() const
{
  return m_CostFunction->GetOptimiseTimingLag();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetTrackingData(mitk::TrackingAndTimeStampsContainer* trackingData)
{
  m_TrackingData = trackingData;
  this->Modified();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetPointData(std::vector< std::pair<unsigned long long, cv::Point3d> >* pointData)
{
  m_PointData = pointData;
  this->Modified();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetRigidTransformation(const cv::Matx44d& rigidBodyTrans)
{
  cv::Matx33d rotationMatrix;
  cv::Matx31d rotationVector;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      rotationMatrix(i,j) = rigidBodyTrans(i,j);
    }
  }
  cv::Rodrigues(rotationMatrix, rotationVector);

  m_RigidTransformation.clear();
  m_RigidTransformation.push_back(rotationVector(0,0));
  m_RigidTransformation.push_back(rotationVector(1,0));
  m_RigidTransformation.push_back(rotationVector(2,0));
  m_RigidTransformation.push_back(rigidBodyTrans(0,3));
  m_RigidTransformation.push_back(rigidBodyTrans(1,3));
  m_RigidTransformation.push_back(rigidBodyTrans(2,3));

  this->Modified();
}


//-----------------------------------------------------------------------------
cv::Matx44d InvariantPointCalibration::GetRigidTransformation() const
{
  assert(m_RigidTransformation.size() == 6);

  cv::Matx44d result;
  mitk::MakeIdentity(result);

  cv::Matx33d rotationMatrix;
  cv::Matx31d rotationVector;

  rotationVector(0, 0) = m_RigidTransformation[0];
  rotationVector(1, 0) = m_RigidTransformation[1];
  rotationVector(2, 0) = m_RigidTransformation[2];
  cv::Rodrigues(rotationVector, rotationMatrix);

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      result(i,j) = rotationMatrix(i,j);
    }
    result(i,3) = m_RigidTransformation[i+3];
  }
  return result;
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::LoadRigidTransformation(const std::string& fileName)
{
  if (fileName.size() > 0)
  {
    cv::Matx44d matrix;
    if (!ReadTrackerMatrix(fileName, matrix))
    {
      mitkThrow() << "Failed to load matrix from file:" << fileName << std::endl;
    }
  }
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SaveRigidTransformation(const std::string& fileName)
{
  if (fileName.size() > 0)
  {
    cv::Matx44d matrix = this->GetRigidTransformation();
    if (!SaveTrackerMatrix(fileName, matrix))
    {
      mitkThrow() << "Failed to save matrix in file:" << fileName << std::endl;
    }
  }
}


//-----------------------------------------------------------------------------
} // end namespace
