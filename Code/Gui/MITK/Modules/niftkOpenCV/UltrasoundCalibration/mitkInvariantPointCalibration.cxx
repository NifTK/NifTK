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
void InvariantPointCalibration::SetTimingLag(const double &timeStamp)
{
  m_CostFunction->SetTimingLag(timeStamp);
  this->Modified();
}

//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetAllowableTimingError(const TimeStampsContainer::TimeStamp &timingError)
{
  m_CostFunction->SetAllowableTimingError(timingError);
  this->Modified();
}



//-----------------------------------------------------------------------------
double InvariantPointCalibration::GetTimingLag()
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
  m_CostFunction->SetRigidTransformation(rigidBodyTrans);
  this->Modified();
}


//-----------------------------------------------------------------------------
cv::Matx44d InvariantPointCalibration::GetRigidTransformation() const
{
  return m_CostFunction->GetRigidTransformation();
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
    this->SetRigidTransformation(matrix);
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
void InvariantPointCalibration::SetOptimiseRigidTransformation(const bool& optimise)
{
  m_CostFunction->SetOptimiseRigidTransformation(optimise);
  this->Modified();
}


//-----------------------------------------------------------------------------
bool InvariantPointCalibration::GetOptimiseRigidTransformation() const
{
  return m_CostFunction->GetOptimiseRigidTransformation();
}


//-----------------------------------------------------------------------------
void InvariantPointCalibration::SetVerbose(const bool& verbose)
{
  m_CostFunction->SetVerbose(verbose);
  this->Modified();
}


//-----------------------------------------------------------------------------
bool InvariantPointCalibration::GetVerbose() const
{
  return m_CostFunction->GetVerbose();
}

//-----------------------------------------------------------------------------
} // end namespace
