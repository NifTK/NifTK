/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGITrackerDataType.h"

namespace niftk
{

//-----------------------------------------------------------------------------
IGITrackerDataType::~IGITrackerDataType()
{
}


//-----------------------------------------------------------------------------
IGITrackerDataType::IGITrackerDataType()
{
  m_TrackingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_TrackingMatrix->Identity();
}


//-----------------------------------------------------------------------------
IGITrackerDataType::IGITrackerDataType(const IGITrackerDataType& other)
: IGIDataType(other)
{
  m_ToolName = other.m_ToolName;
  m_TrackingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_TrackingMatrix->DeepCopy(other.m_TrackingMatrix);
}


//-----------------------------------------------------------------------------
IGITrackerDataType::IGITrackerDataType(IGITrackerDataType&& other)
: IGIDataType(other)
{
  m_ToolName = other.m_ToolName;
  m_TrackingMatrix = other.m_TrackingMatrix;
  other.m_TrackingMatrix = nullptr;
}


//-----------------------------------------------------------------------------
IGITrackerDataType& IGITrackerDataType::operator=(const IGITrackerDataType& other)
{
  IGIDataType::operator=(other);
  m_ToolName = other.m_ToolName;
  m_TrackingMatrix->DeepCopy(other.m_TrackingMatrix);
  return *this;
}


//-----------------------------------------------------------------------------
IGITrackerDataType& IGITrackerDataType::operator=(IGITrackerDataType&& other)
{
  IGIDataType::operator=(other);
  m_ToolName = other.m_ToolName;
  m_TrackingMatrix = other.m_TrackingMatrix;
  other.m_TrackingMatrix = nullptr;
  return *this;
}


//-----------------------------------------------------------------------------
void IGITrackerDataType::SetTrackingMatrix(const vtkSmartPointer<vtkMatrix4x4>& data)
{
  m_TrackingMatrix->DeepCopy(data);
}


//-----------------------------------------------------------------------------
void IGITrackerDataType::SetTrackingData(const std::vector<double>& transform)
{
  for (int r = 0; r < 4; r++)
  {
    for (int c = 0; c < 4; c++)
    {
      m_TrackingMatrix->SetElement(r, c, transform[r*4 + c]);
    }
  }
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> IGITrackerDataType::GetTrackingMatrix() const
{
  vtkSmartPointer<vtkMatrix4x4> tmp = vtkSmartPointer<vtkMatrix4x4>::New();
  tmp->DeepCopy(m_TrackingMatrix);
  return tmp;
}


//-----------------------------------------------------------------------------
void IGITrackerDataType::Clone(const IGIDataType& other)
{
  IGIDataType::Clone(other);
  const IGITrackerDataType* tmp = dynamic_cast<const IGITrackerDataType*>(&other);
  if (tmp != nullptr)
  {
    this->SetTrackingMatrix(tmp->m_TrackingMatrix);
  }
  else
  {
    mitkThrow() << "Incorrect data type provided";
  }
}

} // end namespace
