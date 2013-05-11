/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryCrossStructuringElement.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkVotingBinaryIterativeHoleFillingImageFilter.h>
#include <itkLargestConnectedComponentFilter.h>
#include <itkMIDASDownSamplingFilter.h>
#include <itkMIDASUpSamplingFilter.h>

/*!
 * \file niftkExtractScalp.cxx
 * \page niftkExtractScalp
 * \section niftkExtractScalpSummary Implements Dogdas et al. Human Brain Mapping 26:273-285(2005) to extract the scalp.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "Implements Dogdas et al. Human Brain Mapping 26:273-285(2005) to extract the scalp." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -image image -mask brainMask -scalp scalpImage [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -image <filename>       Input image, T1, containing a head" << std::endl;
    std::cout << "    -mask  <filename>       Input brain mask, binary" << std::endl;
    std::cout << "    -scalp <filename>       Output image of scalp" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -tskull <float>         t_skull parameter in paper" << std::endl;
    std::cout << "    -tscalp <float>         t_scalp parameter in paper" << std::endl;
    std::cout << "    -radius <int>      [1]  Radius of structuring elements in dilation/erosion" << std::endl;
    std::cout << "    -fg     <float>    [1]  Foreground value" << std::endl;
    std::cout << "    -bg     <float>    [1]  Background value" << std::endl;
    std::cout << "    -tol    <float>  [0.1]  Tolerance when comparing values to zero" << std::endl;
    std::cout << "    -iters  <int>     [10]  Maximum number of itertions in hole filling filter" << std::endl;
    std::cout << "    -mode   <int>      [0]  Mode: Choose 0 for as per original paper" << std::endl;
    std::cout << "                                  Choose 1 for MIDAS based method" << std::endl << std::endl;
    std::cout << " If mode == 0 " << std::endl;
    std::cout << std::endl;
    std::cout << " If mode == 1 " << std::endl;
    std::cout << "    -subsample <int>   [4]  Subsampling factor" << std::endl;
  }

struct arguments
{
  std::string inputImage;
  std::string inputMask;
  std::string outputScalpImage;
  float tskull;
  float tscalp;
  int radius;
  int maxIterations;
  float foreGround;
  float backGround;
  float tolerance;
  int mode;
  int subsampling;
};

template <int Dimension, class PixelType>
int DoMain(arguments args)
{
  double t_skull = 0;
  double t_scalp = 0;
  unsigned long int voxelCounter = 0;
  PixelType maskPixel;
  PixelType imagePixel;
  PixelType foreGround = (PixelType)(args.foreGround);
  PixelType backGround = (PixelType)(args.backGround);

  typedef typename itk::Image< PixelType, Dimension >     InputImageType;
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  typedef typename itk::BinaryThresholdImageFilter<InputImageType, InputImageType> BinaryThresholdFilterType;
  typedef typename itk::BinaryCrossStructuringElement<PixelType, Dimension> CrossStructuringElementType;
  typedef typename itk::BinaryBallStructuringElement<PixelType, Dimension> BallStructuringElementType;
  typedef typename itk::BinaryDilateImageFilter<InputImageType, InputImageType, CrossStructuringElementType> DilateByCrossImageFilterType;
  typedef typename itk::BinaryDilateImageFilter<InputImageType, InputImageType, BallStructuringElementType> DilateByBallImageFilterType;
  typedef typename itk::BinaryErodeImageFilter<InputImageType, InputImageType, CrossStructuringElementType> ErodeByCrossImageFilterType;
  typedef typename itk::BinaryErodeImageFilter<InputImageType, InputImageType, BallStructuringElementType> ErodeByBallImageFilterType;
  typedef typename itk::VotingBinaryIterativeHoleFillingImageFilter<InputImageType> HoleFillingFilterType;
  typedef typename itk::LargestConnectedComponentFilter<InputImageType, InputImageType> LargestConnectedComponentType;
  typedef typename itk::MIDASDownSamplingFilter<InputImageType, InputImageType> DownSamplingFilterType;
  typedef typename itk::MIDASUpSamplingFilter<InputImageType, InputImageType> UpSamplingFilterType;

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);
  imageReader->Update();

  typename InputImageReaderType::Pointer maskReader = InputImageReaderType::New();
  maskReader->SetFileName(args.inputMask);
  maskReader->Update();

  /** Calculate mean of non-brain voxels, described as t_skull in paper, in equation 3. */
  typename itk::ImageRegionConstIterator<InputImageType> imageIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
  typename itk::ImageRegionConstIterator<InputImageType> maskIterator(maskReader->GetOutput(), maskReader->GetOutput()->GetLargestPossibleRegion());
  for (imageIterator.GoToBegin(), maskIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator, ++maskIterator)
  {
    maskPixel = maskIterator.Get();
    imagePixel = imageIterator.Get();

    if (fabs((double)(imagePixel - 0)) > args.tolerance      // Non-zero voxels
        && fabs((double)(maskPixel - 0)) < args.tolerance)   // That are non-brain
    {
      voxelCounter++;
      t_skull += imagePixel;
    }
  }
  t_skull = t_skull/(double)voxelCounter;
  std::cerr << "Calculated t_skull=" << t_skull << ", using " << voxelCounter << " samples " << std::endl;
  if (args.tskull != std::numeric_limits<float>::min())
  {
    t_skull = args.tskull;
    std::cerr << "Overrode t_skull=" << t_skull << std::endl;
  }

  /** Calculate mean of non-brain voxels that are at or above t_skull. */
  voxelCounter = 0;
  for (imageIterator.GoToBegin(), maskIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator, ++maskIterator)
  {
    maskPixel = maskIterator.Get();
    imagePixel = imageIterator.Get();

    if (fabs((double)(maskPixel - 0)) < args.tolerance  // non-brain voxels
        && imagePixel >= t_skull)
    {
      voxelCounter++;
      t_scalp += imagePixel;
    }
  }
  t_scalp = t_scalp/(double)voxelCounter;
  std::cerr << "Calculated t_scalp=" << t_scalp << ", using " << voxelCounter << " samples " << std::endl;
  if (args.tscalp != std::numeric_limits<float>::min())
  {
    t_scalp = args.tscalp;
    std::cerr << "Overrode t_scalp=" << t_scalp << std::endl;
  }

  typename BinaryThresholdFilterType::Pointer binaryThreshold = BinaryThresholdFilterType::New();
  typename DilateByCrossImageFilterType::Pointer dilateByCross = DilateByCrossImageFilterType::New();
  typename DilateByBallImageFilterType::Pointer dilateByBall = DilateByBallImageFilterType::New();
  typename HoleFillingFilterType::Pointer holeFilling = HoleFillingFilterType::New();
  typename ErodeByCrossImageFilterType::Pointer erodeByCross = ErodeByCrossImageFilterType::New();
  typename ErodeByBallImageFilterType::Pointer erodeByBall = ErodeByBallImageFilterType::New();
  typename LargestConnectedComponentType::Pointer largest = LargestConnectedComponentType::New();
  typename DownSamplingFilterType::Pointer downSampling = DownSamplingFilterType::New();
  typename UpSamplingFilterType::Pointer upSampling = UpSamplingFilterType::New();
  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  CrossStructuringElementType crossElement;
  crossElement.SetRadius(args.radius);
  crossElement.CreateStructuringElement();

  BallStructuringElementType ballElement;
  ballElement.SetRadius(args.radius);
  ballElement.CreateStructuringElement();

  imageWriter->SetFileName(args.outputScalpImage);

  binaryThreshold->SetInput(imageReader->GetOutput());
  binaryThreshold->SetOutsideValue(backGround);
  binaryThreshold->SetInsideValue(foreGround);
  binaryThreshold->SetUpperThreshold(std::numeric_limits<PixelType>::max());
  binaryThreshold->SetLowerThreshold(static_cast<typename InputImageType::PixelType>(t_scalp));

  dilateByBall->SetKernel(ballElement);
  dilateByBall->SetDilateValue(foreGround);
  dilateByBall->SetBackgroundValue(backGround);
  dilateByBall->SetBoundaryToForeground(false);

  dilateByCross->SetKernel(crossElement);
  dilateByCross->SetDilateValue(foreGround);
  dilateByCross->SetBackgroundValue(backGround);
  dilateByCross->SetBoundaryToForeground(false);

  erodeByCross->SetKernel(crossElement);
  erodeByCross->SetErodeValue(foreGround);
  erodeByCross->SetBackgroundValue(backGround);
  erodeByCross->SetBoundaryToForeground(false);

  erodeByBall->SetKernel(ballElement);
  erodeByBall->SetErodeValue(foreGround);
  erodeByBall->SetBackgroundValue(backGround);
  erodeByBall->SetBoundaryToForeground(false);

  holeFilling->SetMaximumNumberOfIterations(args.maxIterations);
  downSampling->SetDownSamplingFactor(args.subsampling);
  upSampling->SetUpSamplingFactor(args.subsampling);

  std::cerr << "Mode=" << args.mode << std::endl;

  if (args.mode == 0)
  {

    dilateByCross->SetInput(binaryThreshold->GetOutput());
    dilateByBall->SetInput(dilateByCross->GetOutput());
    holeFilling->SetInput(dilateByBall->GetOutput());
    erodeByCross->SetInput(holeFilling->GetOutput());
    erodeByBall->SetInput(erodeByCross->GetOutput());
    largest->SetInput(erodeByBall->GetOutput());
    imageWriter->SetInput(largest->GetOutput());

  }
  else
  {

    downSampling->SetInput(binaryThreshold->GetOutput());
    erodeByBall->SetInput(downSampling->GetOutput());
    holeFilling->SetInput(erodeByBall->GetOutput());
    upSampling->SetInput(holeFilling->GetOutput());
    largest->SetInput(upSampling->GetOutput());
    imageWriter->SetInput(downSampling->GetOutput());

  }

  try
  {
    imageWriter->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

/**
 * \brief Extracts the skull inner and outer boundary.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.tskull = std::numeric_limits<float>::min();
  args.tscalp = std::numeric_limits<float>::min();
  args.radius = 1;
  args.maxIterations = 10;
  args.foreGround = 1;
  args.backGround = 0;
  args.tolerance = 0.1;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-image") == 0){
      args.inputImage=argv[++i];
      std::cout << "Set -image=" << args.inputImage << std::endl;
    }
    else if(strcmp(argv[i], "-mask") == 0){
      args.inputMask=argv[++i];
      std::cout << "Set -mask=" << args.inputMask << std::endl;
    }
    else if(strcmp(argv[i], "-scalp") == 0){
      args.outputScalpImage=argv[++i];
      std::cout << "Set -scalp=" << args.outputScalpImage << std::endl;
    }
    else if(strcmp(argv[i], "-tskull") == 0){
      args.tskull=atof(argv[++i]);
      std::cout << "Set -tskull=" << niftk::ConvertToString(args.tskull) << std::endl;
    }
    else if(strcmp(argv[i], "-tscalp") == 0){
      args.tscalp=atof(argv[++i]);
      std::cout << "Set -tscalp=" << niftk::ConvertToString(args.tscalp) << std::endl;
    }
    else if(strcmp(argv[i], "-radius") == 0){
      args.radius=atoi(argv[++i]);
      std::cout << "Set -radius=" << niftk::ConvertToString(args.radius) << std::endl;
    }
    else if(strcmp(argv[i], "-iters") == 0){
      args.maxIterations=atoi(argv[++i]);
      std::cout << "Set -iters=" << niftk::ConvertToString(args.maxIterations) << std::endl;
    }
    else if(strcmp(argv[i], "-fg") == 0){
      args.foreGround=atof(argv[++i]);
      std::cout << "Set -fg=" << niftk::ConvertToString(args.foreGround) << std::endl;
    }
    else if(strcmp(argv[i], "-bg") == 0){
      args.backGround=atof(argv[++i]);
      std::cout << "Set -bg=" << niftk::ConvertToString(args.backGround) << std::endl;
    }
    else if(strcmp(argv[i], "-tol") == 0){
      args.tolerance=atof(argv[++i]);
      std::cout << "Set -tol=" << niftk::ConvertToString(args.tolerance) << std::endl;
    }
    else if(strcmp(argv[i], "-mode") == 0){
      args.mode=atoi(argv[++i]);
      std::cout << "Set -mode=" << niftk::ConvertToString(args.mode) << std::endl;
    }
    else if(strcmp(argv[i], "-subsample") == 0){
      args.subsampling=atoi(argv[++i]);
      std::cout << "Set -subsample=" << niftk::ConvertToString(args.subsampling) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }

  // Validate command line args
  if (args.inputImage.length() == 0 || args.inputMask.length() == 0 || args.outputScalpImage.length() == 0 )
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::UCHAR:
      if (dims == 2)
        {
          result = DoMain<2, unsigned char>(args);
        }
      else
        {
          result = DoMain<3, unsigned char>(args);
        }
      break;
    case itk::ImageIOBase::CHAR:
      if (dims == 2)
        {
          result = DoMain<2, char>(args);
        }
      else
        {
          result = DoMain<3, char>(args);
        }
      break;
    case itk::ImageIOBase::USHORT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned short>(args);
        }
      else
        {
          result = DoMain<3, unsigned short>(args);
        }
      break;
    case itk::ImageIOBase::SHORT:
      if (dims == 2)
        {
          result = DoMain<2, short>(args);
        }
      else
        {
          result = DoMain<3, short>(args);
        }
      break;
    case itk::ImageIOBase::UINT:
      if (dims == 2)
        {
          result = DoMain<2, unsigned int>(args);
        }
      else
        {
          result = DoMain<3, unsigned int>(args);
        }
      break;
    case itk::ImageIOBase::INT:
      if (dims == 2)
        {
          result = DoMain<2, int>(args);
        }
      else
        {
          result = DoMain<3, int>(args);
        }
      break;
    case itk::ImageIOBase::ULONG:
      if (dims == 2)
        {
          result = DoMain<2, unsigned long>(args);
        }
      else
        {
          result = DoMain<3, unsigned long>(args);
        }
      break;
    case itk::ImageIOBase::LONG:
      if (dims == 2)
        {
          result = DoMain<2, long>(args);
        }
      else
        {
          result = DoMain<3, long>(args);
        }
      break;
    case itk::ImageIOBase::FLOAT:
      if (dims == 2)
        {
          result = DoMain<2, int>(args);
        }
      else
        {
          result = DoMain<3, int>(args);
        }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 2)
        {
          result = DoMain<2, int>(args);
        }
      else
        {
          result = DoMain<3, int>(args);
        }
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
