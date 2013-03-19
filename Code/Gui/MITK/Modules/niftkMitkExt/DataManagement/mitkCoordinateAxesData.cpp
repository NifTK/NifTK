/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkCoordinateAxesData.h"
#include <mitkVector.h>
#include <itkMatrix.h>
#include <itkVector.h>

namespace mitk
{

//-----------------------------------------------------------------------------
CoordinateAxesData::CoordinateAxesData()
{
  IndexType i;
  i.Fill(0);

  SizeType s;
  s.Fill(0);

  m_LargestPossibleRegion.SetSize(s);
  m_LargestPossibleRegion.SetIndex(i);
  m_RequestedRegion.SetSize(s);
  m_RequestedRegion.SetIndex(i);
}


//-----------------------------------------------------------------------------
CoordinateAxesData::~CoordinateAxesData()
{
}


//-----------------------------------------------------------------------------
void CoordinateAxesData::SetRequestedRegionToLargestPossibleRegion()
{
  // Deliberately blank, as nothing to do.
}


//-----------------------------------------------------------------------------
bool CoordinateAxesData::RequestedRegionIsOutsideOfTheBufferedRegion()
{
  return false;
}


//-----------------------------------------------------------------------------
bool CoordinateAxesData::VerifyRequestedRegion()
{
  return true;
}


//-----------------------------------------------------------------------------
void CoordinateAxesData::SetRequestedRegion(itk::DataObject *data)
{
  // Deliberately blank, as nothing to do.
}


//-----------------------------------------------------------------------------
const CoordinateAxesData::RegionType& CoordinateAxesData::GetLargestPossibleRegion() const
{
  m_LargestPossibleRegion.SetIndex(3, 0);
  m_LargestPossibleRegion.SetSize(3, GetTimeSlicedGeometry()->GetTimeSteps());
  return m_LargestPossibleRegion;
}


//-----------------------------------------------------------------------------
const CoordinateAxesData::RegionType& CoordinateAxesData::GetRequestedRegion() const
{
  return GetLargestPossibleRegion();
}


//-----------------------------------------------------------------------------
void CoordinateAxesData::GetVtkMatrix(vtkMatrix4x4& matrixToWriteTo) const
{
  mitk::TimeSlicedGeometry::ConstPointer geometry = this->GetTimeSlicedGeometry();
  if (geometry.IsNotNull())
  {
    mitk::AffineTransform3D::ConstPointer itkTrans = geometry->GetIndexToWorldTransform();
    itk::Matrix<mitk::ScalarType, 3,3> matrix = itkTrans->GetMatrix();
    itk::Vector<mitk::ScalarType, 3> offset = itkTrans->GetOffset();

    matrixToWriteTo.Identity();
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        matrixToWriteTo[i][j] = matrix[i][j];
      }
      matrixToWriteTo[i][3] = offset[i];
    }
  }
}


//-----------------------------------------------------------------------------
void CoordinateAxesData::SetVtkMatrix(const vtkMatrix4x4& matrix)
{
  mitk::TimeSlicedGeometry::Pointer geometry = this->GetTimeSlicedGeometry();
  if (geometry.IsNotNull())
  {
    geometry->SetIndexToWorldTransformByVtkMatrix(const_cast<vtkMatrix4x4*>(&matrix));
  }
}

} // end namespace
