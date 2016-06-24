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
IGITrackerDataType::IGITrackerDataType()
{
  m_TrackingData = vtkSmartPointer<vtkMatrix4x4>::New();
  m_TrackingData->Identity();
}


//-----------------------------------------------------------------------------
IGITrackerDataType::~IGITrackerDataType()
{
}


//-----------------------------------------------------------------------------
void IGITrackerDataType::SetTrackingData(vtkSmartPointer<vtkMatrix4x4> data)
{
  m_TrackingData->DeepCopy(data);
  this->Modified();
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> IGITrackerDataType::GetTrackingData() const
{
  vtkSmartPointer<vtkMatrix4x4> tmp = vtkSmartPointer<vtkMatrix4x4>::New();
  tmp->DeepCopy(m_TrackingData);
  return tmp;
}

} // end namespace
