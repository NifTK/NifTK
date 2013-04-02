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
#include "itkMIDASHelper.h"

namespace mitk
{

//-----------------------------------------------------------------------------
itk::ORIENTATION_ENUM GetItkOrientation(const MIDASOrientation& orientation)
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
MIDASOrientation GetMitkOrientation(const itk::ORIENTATION_ENUM& orientation)
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
int GetUpDirection(const mitk::Geometry3D* geometry, const MIDASOrientation& orientation)
{

  int result = 0;
  if (geometry != NULL && orientation != MIDAS_ORIENTATION_UNKNOWN)
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
      itk::ORIENTATION_ENUM itkOrientation = GetItkOrientation(orientation);
      int axisOfInterest = itk::GetAxisFromOrientationString(orientationString, itkOrientation);

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

  itk::ORIENTATION_ENUM itkOrientation = GetItkOrientation(orientation);
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

  itk::ORIENTATION_ENUM itkOrientation = GetItkOrientation(orientation);
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

} // end namespace
