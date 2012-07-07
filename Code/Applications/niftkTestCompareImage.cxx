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

#include <sstream>
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIteratorWithIndex.h"

/*!
 * \file niftkTestCompareImage.cxx
 * \page niftkTestCompareImage
 * \section niftkTestComareImageSummary Simple program, mainly for integration testing to compare 2 images, subject to a tolerance, or certain other assumptions.
 */
void Usage(char *exec)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Checks scalar images. Will throw exceptions if check fails. By default does nothing. You have to specify at least one test to perform." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << exec << " -i inputFileName1 [options]" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i    <filename>         Input image1. " << std::endl;
  std::cout << "*** [options]   ***" << std::endl << std::endl;
  std::cout << "    -j    <filename>         Input image2.  " << std::endl;
  std::cout << "    -tol  <double>           General tolerance applied to double comparisons. Default 0.0001." << std::endl;
  std::cout << "    -intensity               Do voxel-wise intensity comparison. Requires two images. " << std::endl;
  std::cout << "    -min min                 Check the min. Requires one image." << std::endl;
  std::cout << "    -max max                 Check the max. Requires one image." << std::endl;
}

struct arguments
{
  double tolerance;
  std::string inputImage1;
  std::string inputImage2;
  bool doVoxelCheck;
  bool doMinCheck;
  bool doMaxCheck;
  double min;
  double max;
};

template <int Dimension, class PixelType>
void CheckSameNumberOfVoxels(
    typename itk::Image<PixelType, Dimension>* image1,
    typename itk::Image<PixelType, Dimension>* image2
    )
{
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef typename ImageType::SizeType SizeType;

  SizeType image1Size = image1->GetLargestPossibleRegion().GetSize();
  unsigned long int image1Voxels = 1;
  for (unsigned int i = 0; i < image1Size.GetSizeDimension(); i++)
  {
    image1Voxels *= image1Size[i];
  }

  SizeType image2Size = image2->GetLargestPossibleRegion().GetSize();
  unsigned long int image2Voxels = 1;
  for (unsigned int i = 0; i < image2Size.GetSizeDimension(); i++)
  {
    image2Voxels *= image2Size[i];
  }

  if (image1Voxels != image2Voxels)
  {
    std::ostringstream oss;
    oss << "CheckSameNumberOfVoxels failed, image 1 has " << image1Voxels << " voxels whereas image 2 has " << image2Voxels << std::endl;
    throw std::runtime_error(oss.str());
  }
}

template <int Dimension, class PixelType>
void CheckSameSize(
    typename itk::Image<PixelType, Dimension>* image1,
    typename itk::Image<PixelType, Dimension>* image2
    )
{
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef typename ImageType::SizeType SizeType;

  SizeType image1Size = image1->GetLargestPossibleRegion().GetSize();
  SizeType image2Size = image2->GetLargestPossibleRegion().GetSize();

  if (image1Size != image2Size)
  {
    std::ostringstream oss;
    oss << "CheckSameSize failed, image 1 is " << image1Size << " voxels whereas image 2 is " << image2Size << std::endl;
    throw std::runtime_error(oss.str());
  }
}

template <int Dimension, class PixelType>
void CheckSameSpacing(
    typename itk::Image<PixelType, Dimension>* image1,
    typename itk::Image<PixelType, Dimension>* image2
    )
{
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef typename ImageType::SpacingType SpacingType;

  SpacingType image1Spacing = image1->GetSpacing();
  SpacingType image2Spacing = image2->GetSpacing();

  if (image1Spacing != image2Spacing)
  {
    std::ostringstream oss;
    oss << "CheckSameSpacing failed, image 1 has " << image1Spacing << " (mm) whereas image 2 has " << image2Spacing << std::endl;
    throw std::runtime_error(oss.str());
  }
}

template <int Dimension, class PixelType>
void CheckSameOrigin(
    typename itk::Image<PixelType, Dimension>* image1,
    typename itk::Image<PixelType, Dimension>* image2
    )
{
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef typename ImageType::PointType PointType;

  PointType image1Origin = image1->GetOrigin();
  PointType image2Origin = image2->GetOrigin();

  if (image1Origin != image1Origin)
  {
    std::ostringstream oss;
    oss << "CheckSameOrigin failed, image 1 has " << image1Origin << " (mm) whereas image 2 has " << image2Origin << std::endl;
    throw std::runtime_error(oss.str());
  }
}

template <int Dimension, class PixelType>
void CheckSameDirection(
    typename itk::Image<PixelType, Dimension>* image1,
    typename itk::Image<PixelType, Dimension>* image2
    )
{
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef typename ImageType::DirectionType DirectionType;

  DirectionType image1Direction = image1->GetDirection();
  DirectionType image2Direction = image2->GetDirection();

  if (image1Direction != image2Direction)
  {
    std::ostringstream oss;
    oss << "CheckSameDirection failed, image 1 has " << image1Direction << " whereas image 2 has " << image2Direction << std::endl;
    throw std::runtime_error(oss.str());
  }
}

template <int Dimension, class PixelType>
void CompareIntensityValues(
    typename itk::Image<PixelType, Dimension>* image1,
    typename itk::Image<PixelType, Dimension>* image2,
    double tolerance
    )
{
  CheckSameNumberOfVoxels<Dimension, PixelType>(image1, image2);
  CheckSameSize<Dimension, PixelType>(image1, image2);
  CheckSameSpacing<Dimension, PixelType>(image1, image2);
  CheckSameOrigin<Dimension, PixelType>(image1, image2);
  CheckSameDirection<Dimension, PixelType>(image1, image2);

  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef typename ImageType::IndexType IndexType;

  itk::ImageRegionConstIterator<ImageType> iter1(image1, image1->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<ImageType> iter2(image2, image2->GetLargestPossibleRegion());

  for(iter1.GoToBegin(), iter2.GoToBegin(); !iter1.IsAtEnd() && !iter2.IsAtEnd(); ++iter1, ++iter2)
  {
    if (fabs((double)iter1.Get() - (double)iter2.Get()) > tolerance)
    {
      IndexType index1 = iter1.GetIndex();
      IndexType index2 = iter2.GetIndex();

      std::ostringstream oss;
      oss << "CompareIntensityValues failed, image 1 has " << iter1.Get() << " at index " <<  index1 << " whereas image 2 has " << iter2.Get() << " at index " <<  index2 << ", and tolerance = " << tolerance << std::endl;
      throw std::runtime_error(oss.str());
    }
  }

}

template <int Dimension, class PixelType>
void CheckMin(
    typename itk::Image<PixelType, Dimension>* image1,
    double expectedMin,
    double tolerance
    )
{
  typedef itk::Image<PixelType, Dimension> ImageType;
  itk::ImageRegionConstIterator<ImageType> iter(image1, image1->GetLargestPossibleRegion());
  double min = std::numeric_limits<double>::max();

  for (iter.GoToBegin(); !iter.IsAtEnd(); ++ iter)
  {
    if (iter.Get() < min)
    {
      min = iter.Get();
    }
  }

  if (fabs(expectedMin - (double)min) > tolerance)
  {
    std::ostringstream oss;
    oss << "CheckMin failed, image 1 has min of " << min << ", expectedMin was " <<  expectedMin << ", tolerance was " << tolerance << std::endl;
    throw std::runtime_error(oss.str());
  }
}

template <int Dimension, class PixelType>
void CheckMax(
    typename itk::Image<PixelType, Dimension>* image1,
    double expectedMax,
    double tolerance
    )
{
  typedef itk::Image<PixelType, Dimension> ImageType;
  itk::ImageRegionConstIterator<ImageType> iter(image1, image1->GetLargestPossibleRegion());
  double max = std::numeric_limits<double>::min();

  for (iter.GoToBegin(); !iter.IsAtEnd(); ++ iter)
  {
    if (iter.Get() > max)
    {
      max = iter.Get();
    }
  }

  if (fabs(expectedMax - (double)max) > tolerance)
  {
    std::ostringstream oss;
    oss << "CheckMax failed, image 1 has max of " << max << ", expectedMax was " <<  expectedMax << ", tolerance was " << tolerance << std::endl;
    throw std::runtime_error(oss.str());
  }
}

void CheckTwoFileNames(arguments args)
{
  if (args.inputImage1.length() == 0 || args.inputImage2.length() == 0)
  {
    std::ostringstream oss;
    oss << "CheckTwoFileNames failed, filename 1 is " << args.inputImage1 << " and filename 2 is " <<  args.inputImage2 << std::endl;
    throw std::runtime_error(oss.str());
  }
}

template <int Dimension, class PixelType>
int DoMain(arguments args)
{
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;

  typename InputImageReaderType::Pointer imageReader1 = InputImageReaderType::New();
  imageReader1->SetFileName(args.inputImage1);
  imageReader1->Update();

  if (args.doVoxelCheck)
  {
    CheckTwoFileNames(args);

    typename InputImageReaderType::Pointer imageReader2 = InputImageReaderType::New();
    imageReader2->SetFileName(args.inputImage2);
    imageReader2->Update();

    CompareIntensityValues<Dimension, PixelType>(imageReader1->GetOutput(), imageReader2->GetOutput(), args.tolerance);
  }
  if (args.doMinCheck)
  {
    CheckMin<Dimension, PixelType>(imageReader1->GetOutput(), args.min, args.tolerance);
  }
  if (args.doMaxCheck)
  {
    CheckMax<Dimension, PixelType>(imageReader1->GetOutput(), args.max, args.tolerance);
  }
  return EXIT_SUCCESS;
}

/**
 * \brief Compares two images.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.tolerance = 0.0001;
  args.inputImage1 = "";
  args.inputImage2 = "";
  args.doVoxelCheck = false;
  args.doMinCheck = false;
  args.doMaxCheck = false;
  args.min = std::numeric_limits<float>::max();
  args.max = -args.min;

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage1=argv[++i];
      std::cout << "Set -i=" << args.inputImage1<< std::endl;
    }
    else if(strcmp(argv[i], "-j") == 0){
      args.inputImage2=argv[++i];
      std::cout << "Set -j=" << args.inputImage2<< std::endl;
    }
    else if(strcmp(argv[i], "-intensity") == 0)
    {
      args.doVoxelCheck=true;
      std::cout << "Set -intensity=" << niftk::ConvertToString(args.doVoxelCheck) << std::endl;
    }
    else if(strcmp(argv[i], "-tol") == 0)
    {
      args.tolerance = atof(argv[++i]);
      std::cout << "Set -tol=" << niftk::ConvertToString(args.tolerance) << std::endl;
    }
    else if(strcmp(argv[i], "-min") == 0)
    {
      args.doMinCheck = true;
      args.min = atof(argv[++i]);
      std::cout << "Set -min=" << niftk::ConvertToString(args.doMinCheck) \
          << std::endl;
    }
    else if(strcmp(argv[i], "-max") == 0)
    {
      args.doMaxCheck = true;
      args.max = atof(argv[++i]);
      std::cout << "Set -max=" << niftk::ConvertToString(args.doMaxCheck) \
          << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }

  // Validate command line args
  if (args.inputImage1.length() == 0)
    {
      std::cerr << std::endl << "ERROR: At least one filename required." << std::endl << std::endl;
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  if (!args.doVoxelCheck && !args.doMaxCheck && !args.doMinCheck)
    {
      std::cerr << std::endl << "ERROR: At least one test should be specified." << std::endl << std::endl;
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage1);
  if (dims != 2 && dims != 3 && dims != 4)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

  int result;

  switch (itk::PeekAtComponentType(args.inputImage1))
    {
    case itk::ImageIOBase::UCHAR:
      if (dims == 2)
        {
          result = DoMain<2, unsigned char>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, unsigned char>(args);
        }
      else
        {
          result = DoMain<4, unsigned char>(args);
        }
      break;
    case itk::ImageIOBase::CHAR:
      if (dims == 2)
        {
          result = DoMain<2, char>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, char>(args);
        }
      else
        {
          result = DoMain<4, char>(args);
        }
      break;
    case itk::ImageIOBase::USHORT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned short>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, unsigned short>(args);
        }
      else
        {
          result = DoMain<4, unsigned short>(args);
        }
      break;
    case itk::ImageIOBase::SHORT:
      if (dims == 2)
        {
          result = DoMain<2, short>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, short>(args);
        }
      else
        {
          result = DoMain<4, short>(args);
        }
      break;
    case itk::ImageIOBase::UINT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned int>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, unsigned int>(args);
        }
      else
        {
          result = DoMain<4, unsigned int>(args);
        }
      break;
    case itk::ImageIOBase::INT:
      if (dims == 2)
        {
          result = DoMain<2, int>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, int>(args);
        }
      else
        {
          result = DoMain<4, int>(args);
        }
      break;
    case itk::ImageIOBase::ULONG:
      if (dims == 2)
        {
          result = DoMain<2, unsigned long>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, unsigned long>(args);
        }
      else
        {
          result = DoMain<4, unsigned long>(args);
        }
      break;
    case itk::ImageIOBase::LONG:
      if (dims == 2)
        {
          result = DoMain<2, long>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, long>(args);
        }
      else
        {
          result = DoMain<4, long>(args);
        }
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 2)
        {
          result = DoMain<2, float>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, float>(args);
        }
      else
        {
          result = DoMain<4, float>(args);
        }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 2)
        {
          result = DoMain<2, double>(args);
        }
      else if (dims == 3)
        {
          result = DoMain<3, double>(args);
        }
      else
        {
          result = DoMain<4, double>(args);
        }
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
