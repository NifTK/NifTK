/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPointBasedRegistration.h"
#include "mitkFileIOUtils.h"

namespace mitk
{

//-----------------------------------------------------------------------------
PointBasedRegistration::PointBasedRegistration()
{
}


//-----------------------------------------------------------------------------
PointBasedRegistration::~PointBasedRegistration()
{
}


//-----------------------------------------------------------------------------
double PointBasedRegistration::Update(
    const mitk::PointSet::Pointer fixedPointSet,
    const mitk::PointSet::Pointer movingPointSet,
    vtkMatrix4x4& outputTransform) const
{
  double error = 9;

  outputTransform.SetElement(0, 0, 10);



  return error;
}


//-----------------------------------------------------------------------------
bool PointBasedRegistration::SaveToFile(const std::string& fileName, const vtkMatrix4x4& transform) const
{
  bool isSuccessful = false;
  if (fileName.length() > 0)
  {
    isSuccessful = mitk::SaveVtkMatrix4x4ToFile(fileName, transform);
  }
  return isSuccessful;
}


//-----------------------------------------------------------------------------
bool PointBasedRegistration::ApplyToNode(const mitk::DataNode::Pointer& node, vtkMatrix4x4& transform, const bool& makeUndoAble) const
{
  bool isSuccessful = false;

  // Possiby should put the implementation of this method in something more generic like mitk::DataStorageUtils.h

  return isSuccessful;
}


} // end namespace

