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
#include "mitkITKImageImport.h"
#include "itkImage.h"
#include "itkImageRegionConstIterator.h"

namespace mitk
{

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
ITKGetAsAcquiredOrientation(
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
MIDASView GetAsAcquiredView(const MIDASView& defaultView, const mitk::Image* image)
{
  MIDASView view = defaultView;
  if (image != NULL)
  {
    // "As Acquired" means you take the orientation of the XY plane
    // in the original image data, so we switch to ITK to work it out.
    MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;

    if (image->GetDimension() >= 3)
    {
      try
      {
        AccessFixedDimensionByItk_n(image, ITKGetAsAcquiredOrientation, 3, (orientation));
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


//-----------------------------------------------------------------------------
bool IsImage(const mitk::DataNode* node)
{
  bool result = false;
  if (node != NULL && dynamic_cast<mitk::Image*>(node->GetData()))
  {
    result = true;
  }
  return true;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
ITKImagesHaveEqualIntensities(
    const itk::Image<TPixel, VImageDimension>* itkImage,
    const mitk::Image* image2,
    bool &output
    )
{
  output = true;

  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  typename ImageToItkType::Pointer itkImage2 = ImageToItkType::New();
  itkImage2->SetInput(image2);
  itkImage2->Update();

  typename ImageType::ConstPointer im1 = itkImage;
  typename ImageType::ConstPointer im2 = itkImage2->GetOutput();

  itk::ImageRegionConstIterator<ImageType> iter1(im1, im1->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<ImageType> iter2(im2, im2->GetLargestPossibleRegion());
  for (iter1.GoToBegin(), iter2.GoToBegin();
      !iter1.IsAtEnd() && !iter2.IsAtEnd();
      ++iter1, ++iter2
      )
  {
    if (iter1.Get() != iter2.Get())
    {
      output = false;
      return;
    }
  }
}


//-----------------------------------------------------------------------------
bool ImagesHaveEqualIntensities(const mitk::Image* image1, const mitk::Image* image2)
{
  bool result = false;

  if (image1 != NULL && image2 != NULL)
  {
    try
    {
      AccessByItk_n(image1, ITKImagesHaveEqualIntensities, (image2, result));
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "ImagesAreEqual: AccessByItk_n failed to check equality due to." << e.what() << std::endl;
    }
  }

  return result;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
ITKImagesHaveSameSpatialExtent(
    const itk::Image<TPixel, VImageDimension>* itkImage,
    const mitk::Image* image2,
    bool &output
    )
{
  output = true;

  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  typename ImageToItkType::Pointer itkImage2 = ImageToItkType::New();
  itkImage2->SetInput(image2);
  itkImage2->Update();

  typename ImageType::ConstPointer im1 = itkImage;
  typename ImageType::ConstPointer im2 = itkImage2->GetOutput();

  if (im1->GetLargestPossibleRegion() != im2->GetLargestPossibleRegion())
  {
    output = false;
    return;
  }

  if (im1->GetOrigin() != im2->GetOrigin())
  {
    output = false;
    return;
  }

  if (im1->GetSpacing() != im2->GetSpacing())
  {
    output = false;
    return;
  }

  if (im1->GetDirection() != im2->GetDirection())
  {
    output = false;
    return;
  }
}


//-----------------------------------------------------------------------------
bool ImagesHaveSameSpatialExtent(const mitk::Image* image1, const mitk::Image* image2)
{
  bool result = false;

  if (image1 != NULL && image2 != NULL)
  {
    try
    {
      AccessByItk_n(image1, ITKImagesHaveSameSpatialExtent, (image2, result));
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "ImagesAreEqual: AccessByItk_n failed to check equality due to." << e.what() << std::endl;
    }
  }

  return result;
}


} // end namespace

