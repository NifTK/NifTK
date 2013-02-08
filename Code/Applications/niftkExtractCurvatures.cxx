/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkGaussianCurvatureImageFilter.h"
#include "itkMeanCurvatureImageFilter.h"
#include "itkMinimumCurvatureImageFilter.h"
#include "itkMaximumCurvatureImageFilter.h"

/*!
 * \file niftkExtractCurvatures.cxx
 * \page niftkExtractCurvatures
 * \section niftkExtractCurvaturesSummary Computes Gaussian, mean, minimum or maximum curvatures of a scalar image.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Computes Gaussian, mean, minimum or maximum curvatures of a scalar image." << std::endl;
    std::cout << "  Typical usage will be to run on the output signed distance function of a level set." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i     <filename>        Input image " << std::endl;
    std::cout << "And at least one of:" << std::endl;
    std::cout << "    -gauss <filename>        Output Gaussian curvature image " << std::endl;
    std::cout << "    -mean  <filename>        Output Mean curvature image " << std::endl;
    std::cout << "    -min   <filename>        Output Minimum curvature image" << std::endl;
    std::cout << "    -max   <filename>        Output Maximum curvature image" << std::endl << std::endl;

  }

struct arguments
{
  std::string inputImage;
  std::string gaussianImage;
  std::string meanImage;
  std::string minImage;
  std::string maxImage;
};

template <int Dimension, class PixelType>
int DoMain(arguments args)
{
  typedef typename itk::Image< PixelType, Dimension >     ImageType;
  typedef typename itk::ImageFileReader< ImageType > ReaderType;
  typedef typename itk::ImageFileWriter< ImageType > WriterType;
  typedef typename itk::GaussianCurvatureImageFilter<ImageType, ImageType> GaussianCurvatureFilterType;
  typedef typename itk::MeanCurvatureImageFilter<ImageType, ImageType> MeanCurvatureFilterType;
  typedef typename itk::MinimumCurvatureImageFilter<ImageType, ImageType> MinimumCurvatureFilterType;
  typedef typename itk::MaximumCurvatureImageFilter<ImageType, ImageType> MaximumCurvatureFilterType;

  typename ReaderType::Pointer reader = ReaderType::New();
  typename WriterType::Pointer writer = WriterType::New();
  typename GaussianCurvatureFilterType::Pointer gaussianFilter = GaussianCurvatureFilterType::New();
  typename MeanCurvatureFilterType::Pointer meanFilter = MeanCurvatureFilterType::New();
  typename MinimumCurvatureFilterType::Pointer minimumFilter = MinimumCurvatureFilterType::New();
  typename MaximumCurvatureFilterType::Pointer maximumFilter = MaximumCurvatureFilterType::New();

  try
  {

    reader->SetFileName(args.inputImage);
    reader->Update();

    gaussianFilter->SetInput(reader->GetOutput());
    meanFilter->SetInput(reader->GetOutput());

    if (args.gaussianImage.length() != 0)
    {
      writer->SetInput(gaussianFilter->GetOutput());
      writer->SetFileName(args.gaussianImage);
      writer->Update();
    }

    if (args.meanImage.length() != 0)
    {
      writer->SetInput(meanFilter->GetOutput());
      writer->SetFileName(args.meanImage);
      writer->Update();
    }

    if (args.minImage.length() != 0)
    {
      minimumFilter->SetInput(0, gaussianFilter->GetOutput());
      minimumFilter->SetInput(1, meanFilter->GetOutput());
      writer->SetInput(minimumFilter->GetOutput());
      writer->SetFileName(args.minImage);
      writer->Update();
    }

    if (args.maxImage.length() != 0)
    {
      maximumFilter->SetInput(0, gaussianFilter->GetOutput());
      maximumFilter->SetInput(1, meanFilter->GetOutput());
      writer->SetInput(maximumFilter->GetOutput());
      writer->SetFileName(args.maxImage);
      writer->Update();
    }
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

/**
 * \brief Takes an image and calculates various curvatures.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.gaussianImage = "";
  args.meanImage = "";
  args.minImage = "";
  args.maxImage = "";
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage=argv[++i];
      std::cout << "Set -i=" << args.inputImage << std::endl;
    }
    else if(strcmp(argv[i], "-gauss") == 0) {
      args.gaussianImage=argv[++i];
      std::cout << "Set -gauss=" << args.gaussianImage << std::endl;
    }
    else if(strcmp(argv[i], "-mean") == 0) {
      args.meanImage=argv[++i];
      std::cout << "Set -mean=" << args.meanImage << std::endl;
    }
    else if(strcmp(argv[i], "-min") == 0) {
      args.minImage=argv[++i];
      std::cout << "Set -min=" << args.minImage << std::endl;
    }
    else if(strcmp(argv[i], "-max") == 0) {
      args.maxImage=argv[++i];
      std::cout << "Set -max=" << args.maxImage << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }

  // Validate command line args
  if (args.inputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  if (args.gaussianImage.length() == 0 && args.meanImage.length() == 0 && args.minImage.length() == 0 && args.maxImage.length() == 0)
  {
    std::cerr << "You didn't chose which type of curvature." << std::endl;
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimension(args.inputImage);

  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

  int result = 0;

  if (dims == 2)
  {
    result = DoMain<2, float>(args);
  }
  else
  {
    result = DoMain<3, float>(args);
  }
  return result;
}
