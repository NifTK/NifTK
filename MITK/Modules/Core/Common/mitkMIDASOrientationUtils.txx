/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

namespace mitk
{

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void GetAxesInWorldCoordinateOrder(const itk::Image<TPixel, VImageDimension>* itkImage, int axesInWorldCoordinateOrder[3])
{
  itk::GetAxisFromITKImage(itkImage, itk::ORIENTATION_SAGITTAL, axesInWorldCoordinateOrder[0]);
  itk::GetAxisFromITKImage(itkImage, itk::ORIENTATION_CORONAL, axesInWorldCoordinateOrder[1]);
  itk::GetAxisFromITKImage(itkImage, itk::ORIENTATION_AXIAL, axesInWorldCoordinateOrder[2]);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void GetSpacingInWorldCoordinateOrder(const itk::Image<TPixel, VImageDimension>* itkImage, mitk::Vector3D& spacingInWorldCoordinateOrder)
{
  mitk::Vector3D spacing = itkImage->GetSpacing();
  int axesInWorldCoordinateOrder[3];
  mitk::GetAxesInWorldCoordinateOrder(itkImage, axesInWorldCoordinateOrder);
  spacingInWorldCoordinateOrder[0] = spacing[axesInWorldCoordinateOrder[0]];
  spacingInWorldCoordinateOrder[1] = spacing[axesInWorldCoordinateOrder[1]];
  spacingInWorldCoordinateOrder[2] = spacing[axesInWorldCoordinateOrder[2]];
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void GetExtentsInVxInWorldCoordinateOrder(const itk::Image<TPixel, VImageDimension>* itkImage, mitk::Vector3D& extentsInVxInWorldCoordinateOrder)
{
  int axesInWorldCoordinateOrder[3];
  mitk::GetAxesInWorldCoordinateOrder(itkImage, axesInWorldCoordinateOrder);

  mitk::Vector3D extentsInVx = itkImage->GetExtents();
  extentsInVxInWorldCoordinateOrder[0] = extentsInVx[axesInWorldCoordinateOrder[0]];
  extentsInVxInWorldCoordinateOrder[1] = extentsInVx[axesInWorldCoordinateOrder[1]];
  extentsInVxInWorldCoordinateOrder[2] = extentsInVx[axesInWorldCoordinateOrder[2]];
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void GetExtentsInMmInWorldCoordinateOrder(const itk::Image<TPixel, VImageDimension>* itkImage, mitk::Vector3D& extentsInMmInWorldCoordinateOrder)
{
  int axesInWorldCoordinateOrder[3];
  mitk::GetAxesInWorldCoordinateOrder(itkImage, axesInWorldCoordinateOrder);

  mitk::Vector3D extentsInMm = itkImage->GetExtentsInMM();
  extentsInMmInWorldCoordinateOrder[0] = extentsInMm[axesInWorldCoordinateOrder[0]];
  extentsInMmInWorldCoordinateOrder[1] = extentsInMm[axesInWorldCoordinateOrder[1]];
  extentsInMmInWorldCoordinateOrder[2] = extentsInMm[axesInWorldCoordinateOrder[2]];
}

} // end namespace
