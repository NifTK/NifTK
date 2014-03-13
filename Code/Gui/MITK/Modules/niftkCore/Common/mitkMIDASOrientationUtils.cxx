/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASOrientationUtils.h"
#include <mitkImageAccessByItk.h>
#include <mitkITKImageImport.h>

namespace mitk
{

//-----------------------------------------------------------------------------
itk::Orientation GetItkOrientation(const MIDASOrientation& orientation)
{
  if (orientation == MIDAS_ORIENTATION_AXIAL)
  {
    return itk::ORIENTATION_AXIAL;
  }
  else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
  {
    return itk::ORIENTATION_SAGITTAL;
  }
  else if (orientation == MIDAS_ORIENTATION_CORONAL)
  {
    return itk::ORIENTATION_CORONAL;
  }
  else
  {
    return itk::ORIENTATION_UNKNOWN;
  }
}


//-----------------------------------------------------------------------------
MIDASOrientation GetMitkOrientation(const itk::Orientation& orientation)
{
  if (orientation == itk::ORIENTATION_AXIAL)
  {
    return MIDAS_ORIENTATION_AXIAL;
  }
  else if (orientation == itk::ORIENTATION_SAGITTAL)
  {
    return MIDAS_ORIENTATION_SAGITTAL;
  }
  else if (orientation == itk::ORIENTATION_CORONAL)
  {
    return MIDAS_ORIENTATION_CORONAL;
  }
  else
  {
    return MIDAS_ORIENTATION_UNKNOWN;
  }
}


//-----------------------------------------------------------------------------
int GetUpDirection(const mitk::Geometry3D* geometry, itk::Orientation orientation)
{

  int result = 0;
  if (geometry != NULL && orientation != itk::ORIENTATION_UNKNOWN)
  {
    itk::Matrix<double, 3, 3> directionMatrix;
    VnlVector axis0 = geometry->GetMatrixColumn(0);
    VnlVector axis1 = geometry->GetMatrixColumn(1);
    VnlVector axis2 = geometry->GetMatrixColumn(2);

    axis0 = axis0.normalize();
    axis1 = axis1.normalize();
    axis2 = axis2.normalize();

    directionMatrix[0][0] = axis0[0];
    directionMatrix[1][0] = axis0[1];
    directionMatrix[2][0] = axis0[2];
    directionMatrix[0][1] = axis1[0];
    directionMatrix[1][1] = axis1[1];
    directionMatrix[2][1] = axis1[2];
    directionMatrix[0][2] = axis2[0];
    directionMatrix[1][2] = axis2[1];
    directionMatrix[2][2] = axis2[2];

    std::string orientationString;
    itk::GetOrientationString(directionMatrix, orientationString);

    if (orientationString != "UNKNOWN")
    {
      int axisOfInterest = itk::GetAxisFromOrientationString(orientationString, orientation);

      if (axisOfInterest >= 0)
      {
        result = itk::GetUpDirection(orientationString, axisOfInterest);
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
int GetUpDirection(const mitk::Image* image, const MIDASOrientation& orientation)
{
  int result = 0;

  itk::Orientation itkOrientation = GetItkOrientation(orientation);
  if (image != NULL && itkOrientation != itk::ORIENTATION_UNKNOWN)
  {
    try
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 3:
        AccessFixedDimensionByItk_n(image, itk::GetUpDirectionFromITKImage, 3, (itkOrientation, result));
        break;
      default:
        MITK_ERROR << "During GetUpDirection, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "GetUpDirection: AccessFixedDimensionByItk_n failed to calculate up direction due to." << e.what() << std::endl;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
int GetThroughPlaneAxis(const mitk::Image* image, const MIDASOrientation& orientation)
{
  int result = -1;

  itk::Orientation itkOrientation = GetItkOrientation(orientation);
  if (image != NULL && itkOrientation != itk::ORIENTATION_UNKNOWN)
  {
    try
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 3:
        AccessFixedDimensionByItk_n(image, itk::GetAxisFromITKImage, 3, (itkOrientation, result));
        break;
      default:
        MITK_ERROR << "During GetThroughPlaneAxis, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "GetThroughPlaneAxis: AccessFixedDimensionByItk_n failed to calculate up direction due to." << e.what() << std::endl;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
std::string GetOrientationString(const mitk::Image* image)
{
  std::string result = "UNKNOWN";

  if (image != NULL)
  {
    try
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 3:
        AccessFixedDimensionByItk_n(image, itk::GetOrientationStringFromITKImage, 3, (result));
        break;
      default:
        MITK_ERROR << "During GetOrientationString, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "GetOrientationString: AccessFixedDimensionByItk_n failed to retrieve orientation string due to." << e.what() << std::endl;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
void GetAxesInWorldCoordinateOrder(const mitk::Image* mitkImage, int axesInWorldCoordinateOrder[3])
{
  axesInWorldCoordinateOrder[0] = mitk::GetThroughPlaneAxis(mitkImage, MIDAS_ORIENTATION_SAGITTAL);
  axesInWorldCoordinateOrder[1] = mitk::GetThroughPlaneAxis(mitkImage, MIDAS_ORIENTATION_CORONAL);
  axesInWorldCoordinateOrder[2] = mitk::GetThroughPlaneAxis(mitkImage, MIDAS_ORIENTATION_AXIAL);
}


//-----------------------------------------------------------------------------
void GetSpacingInWorldCoordinateOrder(const mitk::Image* mitkImage, mitk::Vector3D& spacingInWorldCoordinateOrder)
{
  int axesInWorldCoordinateOrder[3];
  mitk::GetAxesInWorldCoordinateOrder(mitkImage, axesInWorldCoordinateOrder);

  mitk::Vector3D spacing = mitkImage->GetGeometry()->GetSpacing();
  spacingInWorldCoordinateOrder[0] = spacing[axesInWorldCoordinateOrder[0]];
  spacingInWorldCoordinateOrder[1] = spacing[axesInWorldCoordinateOrder[1]];
  spacingInWorldCoordinateOrder[2] = spacing[axesInWorldCoordinateOrder[2]];
}


//-----------------------------------------------------------------------------
void GetExtentsInVxInWorldCoordinateOrder(const mitk::Image* mitkImage, mitk::Vector3D& extentsInVxInWorldCoordinateOrder)
{
  int axesInWorldCoordinateOrder[3];
  mitk::GetAxesInWorldCoordinateOrder(mitkImage, axesInWorldCoordinateOrder);

  mitk::Geometry3D* geometry = mitkImage->GetGeometry();
  extentsInVxInWorldCoordinateOrder[0] = geometry->GetExtent(axesInWorldCoordinateOrder[0]);
  extentsInVxInWorldCoordinateOrder[1] = geometry->GetExtent(axesInWorldCoordinateOrder[1]);
  extentsInVxInWorldCoordinateOrder[2] = geometry->GetExtent(axesInWorldCoordinateOrder[2]);
}


//-----------------------------------------------------------------------------
void GetExtentsInMmInWorldCoordinateOrder(const mitk::Image* mitkImage, mitk::Vector3D& extentsInMmInWorldCoordinateOrder)
{
  int axesInWorldCoordinateOrder[3];
  mitk::GetAxesInWorldCoordinateOrder(mitkImage, axesInWorldCoordinateOrder);

  mitk::Geometry3D* geometry = mitkImage->GetGeometry();
  extentsInMmInWorldCoordinateOrder[0] = geometry->GetExtentInMM(axesInWorldCoordinateOrder[0]);
  extentsInMmInWorldCoordinateOrder[1] = geometry->GetExtentInMM(axesInWorldCoordinateOrder[1]);
  extentsInMmInWorldCoordinateOrder[2] = geometry->GetExtentInMM(axesInWorldCoordinateOrder[2]);
}

} // end namespace
