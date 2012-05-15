/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 7828 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkTwinThresholdBoundaryFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <algorithm>

/*!
 * \file niftkCTEAssignAtlasValues.cxx
 * \page niftkCTEAssignAtlasValues
 * \section niftkCTEAssignAtlasValuesSummary Takes an atlas, and an input image, and for each voxel in the input image that is not background will find the nearest atlas voxel, and assign to the output voxel that atlas label.
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Takes an atlas, and an input image, and for each voxel in the input image that is not background" << std::endl;
  std::cout << "  will find the nearest atlas voxel, and assign to the output voxel that atlas label." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -a <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i       <filename>        Input data image." << std::endl;
  std::cout << "    -a       <filename>        Input atlas image." << std::endl;
  std::cout << "    -o       <filename>        Output labelled image." << std::endl << std::endl;
  std::cout << "*** [options]   ***" << std::endl << std::endl;
  std::cout << "    -radius  <float> [5]       Radius in mm over which to search" << std::endl;
  std::cout << "    -bgAtlas <int>   [0]       Background value in atlas image" << std::endl;
  std::cout << "    -bgData  <float> [0]       Background value in data image" << std::endl;
  std::cout << "    -fgData  <float>           If specified, will only search for this foreground value, the default is to do anything thats not background" << std::endl;
}

struct arguments
{
  std::string inputImage;
  std::string atlasImage;
  std::string outputImage;
  float radius;
  float backgroundData;
  float foregroundData;
  int backgroundAtlas;
  bool useForegroundValue;
};

template <int Dimension>
int DoMain(arguments args)
{
  typedef float DataPixelType;
  typedef int MaskPixelType;

  typedef typename itk::Image< DataPixelType, Dimension >      InputDataImageType;
  typedef typename itk::Image< MaskPixelType, Dimension >      InputMaskImageType;
  typedef typename itk::Image< MaskPixelType, Dimension >      OutputImageType;
  typedef typename itk::ImageFileReader< InputDataImageType  > InputDataImageReaderType;
  typedef typename itk::ImageFileReader< InputMaskImageType  > InputMaskImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType >     OutputImageWriterType;

  typename InputDataImageReaderType::Pointer dataReader  = InputDataImageReaderType::New();
  dataReader->SetFileName(  args.inputImage );

  typename InputMaskImageReaderType::Pointer atlasReader  = InputMaskImageReaderType::New();
  atlasReader->SetFileName(  args.atlasImage );
  

  try
    {
      std::cout << "Loading data image:" << args.inputImage << std::endl;
      dataReader->Update();
      std::cout << "Done" << std::endl;

      std::cout << "Loading atlas image:" << args.atlasImage << std::endl;
      atlasReader->Update();
      std::cout << "Done" << std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl;
      return -2;
    }

  typename OutputImageType::Pointer outputImage = OutputImageType::New();
  outputImage->SetRegions(dataReader->GetOutput()->GetLargestPossibleRegion());
  outputImage->SetSpacing(dataReader->GetOutput()->GetSpacing());
  outputImage->SetOrigin(dataReader->GetOutput()->GetOrigin());
  outputImage->SetDirection(dataReader->GetOutput()->GetDirection());
  outputImage->Allocate();
  outputImage->FillBuffer(0);

  // Calculate a region that includes the specified radius.

  typedef typename InputDataImageType::SizeType InputImageSizeType;
  typedef typename InputDataImageType::IndexType InputImageIndexType;
  typedef typename InputDataImageType::SpacingType InputImageSpacingType;
  typedef typename InputDataImageType::RegionType InputImageRegionType;

  InputImageSpacingType imageSpacing = dataReader->GetOutput()->GetSpacing();
  InputImageSizeType windowSize;

  for (unsigned int i = 0; i < Dimension; i++)
    {
      windowSize[i] = (long unsigned int)(((args.radius/imageSpacing[i])*2.0) + 1);
      std::cout << "Data image spacing[" << niftk::ConvertToString((int)i) << "]=" << niftk::ConvertToString((double)imageSpacing[i]) << ", so window size=" << niftk::ConvertToString((int)windowSize[i]) << std::endl;
    }

  itk::ImageRegionConstIteratorWithIndex<InputDataImageType> dataImageIterator(dataReader->GetOutput(), dataReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<OutputImageType> outputImageIterator(outputImage, outputImage->GetLargestPossibleRegion());

  unsigned int i = 0;
  unsigned int count = 0;
  double distance = 0;
  double smallestDistance = 0;
  typename InputMaskImageType::IndexType dataIndex;         dataIndex.SetIndex(0);
  typename InputMaskImageType::IndexType regionIndex;       regionIndex.SetIndex(0);
  typename InputMaskImageType::IndexType closestAtlasIndex; closestAtlasIndex.SetIndex(0);

  for (dataImageIterator.GoToBegin(),
       outputImageIterator.GoToBegin();
       !dataImageIterator.IsAtEnd();
       ++dataImageIterator,
       ++outputImageIterator
       )
    {
      if (   (args.useForegroundValue && dataImageIterator.Get() == args.foregroundData)
          || (!args.useForegroundValue && dataImageIterator.Get() != args.backgroundData)
          )
        {
          dataIndex = dataImageIterator.GetIndex();

          for (i = 0; i < Dimension; i++)
            {
              regionIndex[i] = dataIndex[i] - ((windowSize[i]-1)/2);
            }

          InputImageRegionType region;
          region.SetSize(windowSize);
          region.SetIndex(regionIndex);

          itk::ImageRegionConstIteratorWithIndex<InputDataImageType> dataRegionIterator(dataReader->GetOutput(), region);
          itk::ImageRegionConstIteratorWithIndex<InputMaskImageType> atlasRegionIterator(atlasReader->GetOutput(), region);

          smallestDistance = std::numeric_limits<double>::max();
          count = 0;

          for (atlasRegionIterator.GoToBegin(),
               dataRegionIterator.GoToBegin();
               !atlasRegionIterator.IsAtEnd();
               ++atlasRegionIterator,
               ++dataRegionIterator)
            {
              if (atlasRegionIterator.Get() != args.backgroundAtlas)
              {
                distance = 0;
                regionIndex = atlasRegionIterator.GetIndex();

                for (i = 0; i < Dimension; i++)
                  {
                    distance += (((regionIndex[i] - dataIndex[i]) * imageSpacing[i])
                              *((regionIndex[i] - dataIndex[i]) * imageSpacing[i]));
                  }
                distance = sqrt(distance);

                if (distance <= args.radius && distance < smallestDistance)
                  {
                    smallestDistance = distance;
                    closestAtlasIndex = regionIndex;
                    count++;
                  }
              } // end if not background atlas value
            } // end for

          if (count > 0)
          {
            outputImageIterator.Set(atlasReader->GetOutput()->GetPixel(closestAtlasIndex));
          }
          else
          {
            std::cerr << "WARNING:Failed to find atlas value for index=" << dataIndex << " within radius of " << args.radius << std::endl;
          }
        } // end if mask value != background
    } // end for each voxel

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput(outputImage);

  try
  {
    std::cout << "Saving output image:" << args.outputImage << std::endl;
    imageWriter->Update();
    std::cout << "Done" << std::endl;
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/**
 * \brief Implements the smoothing mentioned in section 2.5 of  Acosta et. al. MIA 13 (2009) 730-743 doi:10.1016/j.media.2009.07.03
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.radius = 5;
  args.backgroundData = 0;
  args.foregroundData = 1;
  args.backgroundAtlas = 0;
  args.useForegroundValue = false;
  

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
    else if(strcmp(argv[i], "-o") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -o=" << args.outputImage << std::endl;
    }
    else if(strcmp(argv[i], "-a") == 0){
      args.atlasImage=argv[++i];
      std::cout << "Set -a=" << args.atlasImage << std::endl;
    }
    else if(strcmp(argv[i], "-atlas") == 0){
      args.atlasImage=argv[++i];
      std::cout << "Set -atlas=" << args.atlasImage << std::endl;
    }
    else if(strcmp(argv[i], "-radius") == 0){
      args.radius=atof(argv[++i]);
      std::cout << "Set -radius=" << niftk::ConvertToString(args.radius) << std::endl;
    }
    else if(strcmp(argv[i], "-bgAtlas") == 0){
      args.backgroundAtlas=atoi(argv[++i]);
      std::cout << "Set -bgAtlas=" << niftk::ConvertToString(args.backgroundAtlas) << std::endl;
    }
    else if(strcmp(argv[i], "-bgData") == 0){
      args.backgroundData=atof(argv[++i]);
      std::cout << "Set -bgData=" << niftk::ConvertToString(args.backgroundData) << std::endl;
    }
    else if(strcmp(argv[i], "-fgData") == 0){
      args.foregroundData=atof(argv[++i]);
      args.useForegroundValue = true;
      std::cout << "Set -fgData=" << niftk::ConvertToString(args.foregroundData) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  if(args.radius < 1 ){
    std::cerr << argv[0] << "\tThe radius must be >= 1" << std::endl;
    return -1;
  }

  int dims = itk::PeekAtImageDimension(args.inputImage);
  int result;

  switch ( dims )
    {
      case 2:
        result = DoMain<2>(args);
        break;
      case 3:
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;
}
