/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASImageUtils.h"
#include "mitkImageAccessByItk.h"

namespace mitk
{

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
GetAsAcquiredOrientation(
  itk::Image<TPixel, VImageDimension>* itkImage,
  MIDASOrientation &outputOrientation
)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;

  typename itk::SpatialOrientationAdapter adaptor;
  typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientation;
  orientation = adaptor.FromDirectionCosines(itkImage->GetDirection());
  std::string orientationString = itk::ConvertSpatialOrientationToString(orientation);

  if (orientationString[0] == 'L' || orientationString[0] == 'R')
  {
    if (orientationString[1] == 'A' || orientationString[1] == 'P')
    {
      outputOrientation = MIDAS_ORIENTATION_AXIAL;
    }
    else
    {
      outputOrientation = MIDAS_ORIENTATION_CORONAL;
    }
  }
  else if (orientationString[0] == 'A' || orientationString[0] == 'P')
  {
    if (orientationString[1] == 'L' || orientationString[1] == 'R')
    {
      outputOrientation = MIDAS_ORIENTATION_AXIAL;
    }
    else
    {
      outputOrientation = MIDAS_ORIENTATION_SAGITTAL;
    }
  }
  else if (orientationString[0] == 'S' || orientationString[0] == 'I')
  {
    if (orientationString[1] == 'L' || orientationString[1] == 'R')
    {
      outputOrientation = MIDAS_ORIENTATION_CORONAL;
    }
    else
    {
      outputOrientation = MIDAS_ORIENTATION_SAGITTAL;
    }
  }
}


//-----------------------------------------------------------------------------
MIDASView GetAsAcquiredView(const MIDASView& defaultView, const mitk::DataNode* node)
{
  MIDASView view = defaultView;
  if (node != NULL)
  {
    // "As Acquired" means you take the orientation of the XY plane
    // in the original image data, so we switch to ITK to work it out.
    MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;

    mitk::Image::Pointer image = NULL;
    image = dynamic_cast<mitk::Image*>(node->GetData());

    if (image.IsNotNull() && image->GetDimension() >= 3)
    {
      try
      {
        AccessFixedDimensionByItk_n(image, GetAsAcquiredOrientation, 3, (orientation));
      }
      catch (const mitk::AccessByItkException &e)
      {
        MITK_ERROR << "GetAsAcquiredView: AccessFixedDimensionByItk_n failed to work out 'As Acquired' orientation." << e.what() << std::endl;
      }
    }
    else
    {
      MITK_ERROR << "GetAsAcquiredView: failed to find an image to work out 'As Acquired' orientation." << std::endl;
    }

    if (orientation == MIDAS_ORIENTATION_AXIAL)
    {
      view = MIDAS_VIEW_AXIAL;
    }
    else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      view = MIDAS_VIEW_SAGITTAL;
    }
    else if (orientation == MIDAS_ORIENTATION_CORONAL)
    {
      view = MIDAS_VIEW_CORONAL;
    }
    else
    {
      MITK_ERROR << "GetAsAcquiredView defaulting to view=" << view << std::endl;
    }
  }
  return view;
}

} // end namespace

