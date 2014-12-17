/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASImageUtils.h"
#include <mitkImageAccessByItk.h>
#include <mitkITKImageImport.h>
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <itkMIDASHelper.h>

namespace mitk
{

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
ITKGetAsAcquiredOrientation(
  const itk::Image<TPixel, VImageDimension>* itkImage,
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
MIDASOrientation GetAsAcquiredOrientation(const MIDASOrientation& defaultOrientation, const mitk::Image* image)
{
  MIDASOrientation result = defaultOrientation;
  if (image != NULL)
  {
    // "As Acquired" means you take the orientation of the XY plane
    // in the original image data, so we switch to ITK to work it out.
    MIDASOrientation orientation = MIDAS_ORIENTATION_UNKNOWN;
    int dimensions = image->GetDimension();
    switch(dimensions)
    {
    case 3:
      AccessFixedDimensionByItk_n(image, ITKGetAsAcquiredOrientation, 3, (orientation));
      break;
    default:
      MITK_ERROR << "During GetAsAcquiredOrientation, unsupported number of dimensions:" << dimensions << std::endl;
    }

    if (orientation == MIDAS_ORIENTATION_AXIAL)
    {
      result = MIDAS_ORIENTATION_AXIAL;
    }
    else if (orientation == MIDAS_ORIENTATION_SAGITTAL)
    {
      result = MIDAS_ORIENTATION_SAGITTAL;
    }
    else if (orientation == MIDAS_ORIENTATION_CORONAL)
    {
      result = MIDAS_ORIENTATION_CORONAL;
    }
    else
    {
      MITK_ERROR << "GetAsAcquiredResult defaulting to orientation=" << result << std::endl;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
bool IsImage(const mitk::DataNode* node)
{
  bool result = false;
  if (node != NULL && dynamic_cast<mitk::Image*>(node->GetData()))
  {
    result = true;
  }
  return result;
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
  itkImage2->SetInput(const_cast<mitk::Image*>(image2));
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
      int dimensions = image1->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessFixedDimensionByItk_n(image1, ITKImagesHaveEqualIntensities, 2, (image2, result));
        break;
      case 3:
        AccessFixedDimensionByItk_n(image1, ITKImagesHaveEqualIntensities, 3, (image2, result));
        break;
      case 4:
        AccessFixedDimensionByItk_n(image1, ITKImagesHaveEqualIntensities, 4, (image2, result));
        break;
      default:
        MITK_ERROR << "During ImagesHaveEqualIntensities, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "ImagesHaveEqualIntensities: AccessFixedDimensionByItk_n failed to check equality due to." << e.what() << std::endl;
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
  itkImage2->SetInput(const_cast<mitk::Image*>(image2));
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
      int dimensions = image1->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessFixedDimensionByItk_n(image1, ITKImagesHaveSameSpatialExtent, 2, (image2, result));
        break;
      case 3:
        AccessFixedDimensionByItk_n(image1, ITKImagesHaveSameSpatialExtent, 3, (image2, result));
        break;
      case 4:
        AccessFixedDimensionByItk_n(image1, ITKImagesHaveSameSpatialExtent, 4, (image2, result));
        break;
      default:
        MITK_ERROR << "During ImagesHaveSameSpatialExtent, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "ImagesHaveSameSpatialExtent: AccessFixedDimensionByItk_n failed to check equality due to." << e.what() << std::endl;
    }
  }

  return result;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
ITKFillImage(
    itk::Image<TPixel, VImageDimension>* itkImage,
    float &value
    )
{
  itkImage->FillBuffer((TPixel)value);
}


//-----------------------------------------------------------------------------
void FillImage(mitk::Image* image, float value)
{
  if (image != NULL)
  {
    try
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessFixedDimensionByItk_n(image, ITKFillImage, 2, (value));
        break;
      case 3:
        AccessFixedDimensionByItk_n(image, ITKFillImage, 3, (value));
        break;
      case 4:
        AccessFixedDimensionByItk_n(image, ITKFillImage, 4, (value));
        break;
      default:
        MITK_ERROR << "During FillImage, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "FillImage: AccessFixedDimensionByItk_n failed to fill images due to." << e.what() << std::endl;
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
ITKCountBetweenThreshold(
    const itk::Image<TPixel, VImageDimension>* itkImage,
    const float &lower,
    const float &upper,
    unsigned long int &outputCount
    )
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  itk::ImageRegionConstIterator<ImageType> iter(itkImage, itkImage->GetLargestPossibleRegion());
  outputCount = 0;
  TPixel value;
  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
  {
    value = iter.Get();
    if (value >= lower && value <= upper)
    {
      outputCount++;
    }
  }
}


//-----------------------------------------------------------------------------
/**
 * \brief Simply iterates through a whole image, counting how many intensity values are >= lower and <= upper.
 * \param lower A lower threshold for intensity values
 * \param upper An upper threshold for intensity values
 * \return unsigned long int The number of voxels.
 */
NIFTKCORE_EXPORT unsigned long int CountBetweenThreshold(const mitk::Image* image, const float& lower, const float& upper)
{
  unsigned long int counter = 0;

  if (image != NULL)
  {
    try
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessFixedDimensionByItk_n(image, ITKCountBetweenThreshold, 2, (lower, upper, counter));
        break;
      case 3:
        AccessFixedDimensionByItk_n(image, ITKCountBetweenThreshold, 3, (lower, upper, counter));
        break;
      case 4:
        AccessFixedDimensionByItk_n(image, ITKCountBetweenThreshold, 4, (lower, upper, counter));
        break;
      default:
        MITK_ERROR << "During CountBetweenThreshold, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "CountBetweenThreshold: AccessFixedDimensionByItk_n failed to count voxels due to." << e.what() << std::endl;
    }
  }

  return counter;
}


//-----------------------------------------------------------------------------
unsigned long int GetNumberOfVoxels(const mitk::Image* image)
{
  unsigned long int counter = 0;

  if (image != NULL)
  {
    counter = 1;
    for (unsigned int i = 0; i < image->GetDimension(); i++)
    {
      counter *= image->GetDimension(i);
    }
  }
  return counter;
}


//-----------------------------------------------------------------------------
mitk::Point3D GetMiddlePointInVoxels(const mitk::Image* image)
{
  mitk::Point3D voxelIndex;
  voxelIndex[0] = (int)(image->GetDimension(0)/2.0);
  voxelIndex[1] = (int)(image->GetDimension(1)/2.0);
  voxelIndex[2] = (int)(image->GetDimension(2)/2.0);
  return voxelIndex;
}


//-----------------------------------------------------------------------------
double GetVolume(const mitk::Image* image)
{
  double volume = 0;

  if (image != NULL)
  {
    try
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessFixedDimensionByItk_n(image, itk::GetVolumeFromITKImage, 2, (volume));
        break;
      case 3:
        AccessFixedDimensionByItk_n(image, itk::GetVolumeFromITKImage, 3, (volume));
        break;
      case 4:
        AccessFixedDimensionByItk_n(image, itk::GetVolumeFromITKImage, 4, (volume));
        break;
      default:
        MITK_ERROR << "During GetVolume, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "GetVolume: AccessFixedDimensionByItk_n failed to calculate volume due to." << e.what() << std::endl;
    }
  }

  return volume;
}


//-----------------------------------------------------------------------------
void UpdateVolumeProperty(const mitk::Image* image, mitk::DataNode* node)
{
  if (image != NULL && node != NULL)
  {
    double volume = GetVolume(image);
    node->SetFloatProperty("midas.volume", (float)volume);
  }
}


//-----------------------------------------------------------------------------
template <typename TPixel1, unsigned int VImageDimension1, typename TPixel2, unsigned int VImageDimension2>
void ITKCopyIntensityData(const itk::Image<TPixel1, VImageDimension1>* input,
                          itk::Image<TPixel2, VImageDimension2>* output
                         )
{
  typedef typename itk::Image<TPixel1, VImageDimension1> ImageType1;
  typedef typename itk::Image<TPixel2, VImageDimension2> ImageType2;

  itk::ImageRegionConstIterator<ImageType1> inputIter(input, input->GetLargestPossibleRegion());
  itk::ImageRegionIterator<ImageType2> outputIter(output, output->GetLargestPossibleRegion());

  for (inputIter.GoToBegin(), outputIter.GoToBegin(); !inputIter.IsAtEnd() && !outputIter.IsAtEnd(); ++inputIter, ++outputIter)
  {
    outputIter.Set(static_cast<TPixel2>(inputIter.Get()));
  }
}


//-----------------------------------------------------------------------------
void CopyIntensityData(const mitk::Image* input, mitk::Image* output)
{
  if (input != NULL && output != NULL)
  {
    try
    {
      int dimensions = input->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessTwoImagesFixedDimensionByItk(input, output, ITKCopyIntensityData, 2);
        break;
      case 3:
        /// Note:
        /// This line should be used here, but it applies write lock on both images,
        /// regardless the constness. Therefore, until this is fixed in the upstream,
        /// I copied the expanded macro definition here, and replaced the 'nonConstImage1'
        /// variable by 'input' so that the const argument version of SetInput is called
        /// and it applies read lock on the image.
        //AccessTwoImagesFixedDimensionByItk(input, output, ITKCopyIntensityData, 3);
        {
          const mitk::PixelType& pixelType1 = input->GetPixelType();
          const mitk::PixelType& pixelType2 = output->GetPixelType();
          const mitk::Image* constImage1 = input;
          const mitk::Image* constImage2 = output;
          mitk::Image* nonConstImage1 = const_cast<mitk::Image*>(constImage1);
          mitk::Image* nonConstImage2 = const_cast<mitk::Image*>(constImage2);
          nonConstImage1->Update();
          nonConstImage2->Update();
          _checkSpecificDimension(input, (3));
          _checkSpecificDimension(output, (3));
          if (pixelType1 == mitk::MakePixelType< itk::Image<unsigned char,3u> >() &&
              pixelType2 == mitk::MakePixelType< itk::Image<unsigned char,3u> >() &&
              constImage1->GetDimension() == 3u && constImage2->GetDimension() == 3u)
          {
            typedef itk::Image<unsigned char,3u> ImageType1;
            typedef itk::Image<unsigned char,3u> ImageType2;
            typedef mitk::ImageToItk<ImageType1> ImageToItkType1;
            typedef mitk::ImageToItk<ImageType2> ImageToItkType2;
            itk::SmartPointer<ImageToItkType1> imagetoitk1 = ImageToItkType1::New();
            imagetoitk1->SetInput(input);
            imagetoitk1->Update();
            itk::SmartPointer<ImageToItkType2> imagetoitk2 = ImageToItkType2::New();
            imagetoitk2->SetInput(nonConstImage2);
            imagetoitk2->Update();
            ITKCopyIntensityData(imagetoitk1->GetOutput(), imagetoitk2->GetOutput());
          } else
          {
            std::string msg("Pixel type ");
            msg.append(pixelType1.GetComponentTypeAsString() );
            msg.append(" or pixel type ");
            msg.append(pixelType2.GetComponentTypeAsString() );
            throw mitk::AccessByItkException(msg);
          }
        }
        break;
      case 4:
        AccessTwoImagesFixedDimensionByItk(input, output, ITKCopyIntensityData, 4);
        break;
      default:
        MITK_ERROR << "During CopyIntensityData, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "CopyIntensityData: AccessTwoImagesFixedDimensionByItk failed to copy data due to." << e.what() << std::endl;
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
ITKDumpImage(
    const itk::Image<TPixel, VImageDimension>* itkImage,
    const std::string& filename
    )
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef itk::ImageFileWriter<ImageType> FileWriterType;

  typename FileWriterType::Pointer writer = FileWriterType::New();
  writer->SetInput(itkImage);
  writer->SetFileName(filename);
  writer->Update();
}


//-----------------------------------------------------------------------------
void DumpImage(const mitk::Image *image, const std::string& fileName)
{
  if (image != NULL)
  {
    try
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessFixedDimensionByItk_n(image, ITKDumpImage, 2, (fileName));
        break;
      case 3:
        AccessFixedDimensionByItk_n(image, ITKDumpImage, 3, (fileName));
        break;
      case 4:
        AccessFixedDimensionByItk_n(image, ITKDumpImage, 4, (fileName));
        break;
      default:
        MITK_ERROR << "During DumpImage, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    catch (const mitk::AccessByItkException &e)
    {
      MITK_ERROR << "FillImage: AccessFixedDimensionByItk_n failed to dump image due to." << e.what() << std::endl;
    }
  }
}



//-----------------------------------------------------------------------------
mitk::Vector3D GetXYAspectRatio(const mitk::Image::Pointer image)
{
  mitk::Vector3D spacing = image->GetGeometry()->GetSpacing();
  if (spacing[1] > spacing[0])
  {
    spacing[1] = spacing[1]/spacing[0];
    spacing[0] = 1;
  }
  else
  {
    spacing[0] = spacing[0]/spacing[1];
    spacing[1] = 1;
  }
  return spacing;
}

} // end namespace

