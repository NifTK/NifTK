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
 * \file niftkCTEMaskedSmoothing.cxx
 * \page niftkCTEMaskedSmoothing
 * \section niftkCTEMaskedSmoothingSummary Takes a data image (eg. thickness) and a binary mask, and for each voxel in the mask image > 0 will compute the inter quartile mean of the data image within a circular radius.
 */
void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Takes a data image (eg. thickness) and a binary mask, and for each voxel in the mask image > 0" << std::endl;
  std::cout << "  will compute the inter quartile mean of the data image within a circular radius." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  Additionally, you can specify an atlas, then for each mask pixel, the value of the atlas at the same pixel" << std::endl;
  std::cout << "  is taken, and then when computing the average over a circular radius, will only include voxels with the same atlas label" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -m <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i       <filename>        Input data image." << std::endl;
  std::cout << "    -m       <filename>        Input mask image." << std::endl;
  std::cout << "    -o       <filename>        Output smoothed image." << std::endl << std::endl;
  std::cout << "*** [options]   ***" << std::endl << std::endl;
  std::cout << "    -radius  <float> [5]       Radius in mm over which to smooth" << std::endl;
  std::cout << "    -atlas   <filename>        Atlas image" << std::endl;
  std::cout << "    -bgData  <float> [0]       Background value in data image" << std::endl;
  std::cout << "    -bgMask  <int>   [0]       Background value in mask image" << std::endl;
  std::cout << "    -bgAtlas <int>   [0]       Background value in atlas image" << std::endl;
  std::cout << "    -justMask                  Only take average over voxels that intersect with mask (which makes -atlas redundant aswell)" << std::endl;
  std::cout << "    -mean                      Do mean, instead of inter-quartile mean" << std::endl;
}

struct arguments
{
  std::string inputImage;
  std::string maskImage;
  std::string outputImage;
  std::string atlasImage;
  float radius;
  float backgroundData;
  int backgroundMask;
  int backgroundAtlas;
  bool justMask;
  bool doMean;
};

template <int Dimension>
int DoMain(arguments args)
{
  typedef float DataPixelType;
  typedef int MaskPixelType;

  typedef typename itk::Image< DataPixelType, Dimension >      InputDataImageType;
  typedef typename itk::Image< MaskPixelType, Dimension >      InputMaskImageType;
  typedef typename itk::Image< DataPixelType, Dimension >      OutputImageType;
  typedef typename itk::ImageFileReader< InputDataImageType  > InputDataImageReaderType;
  typedef typename itk::ImageFileReader< InputMaskImageType  > InputMaskImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType >     OutputImageWriterType;

  typename InputDataImageReaderType::Pointer dataReader  = InputDataImageReaderType::New();
  dataReader->SetFileName(  args.inputImage );

  typename InputMaskImageReaderType::Pointer maskReader  = InputMaskImageReaderType::New();
  maskReader->SetFileName(  args.maskImage );

  typename InputMaskImageReaderType::Pointer atlasReader  = InputMaskImageReaderType::New();
  atlasReader->SetFileName(  args.atlasImage );
  

  try
    {
      std::cout << "Loading data image:" << args.inputImage << std::endl;
      dataReader->Update();
      std::cout << "Done" << std::endl;

      std::cout << "Loading mask image:" << args.maskImage << std::endl;
      maskReader->Update();
      std::cout << "Done" << std::endl;

      if (args.atlasImage.length() > 0)
      {
        std::cout << "Loading atlas image:" << args.atlasImage << std::endl;
        atlasReader->Update();
        std::cout << "Done" << std::endl;
      }
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl;
      return -2;
    }

  typename OutputImageType::Pointer outputImage = OutputImageType::New();
  outputImage->SetRegions(maskReader->GetOutput()->GetLargestPossibleRegion());
  outputImage->SetSpacing(maskReader->GetOutput()->GetSpacing());
  outputImage->SetOrigin(maskReader->GetOutput()->GetOrigin());
  outputImage->SetDirection(maskReader->GetOutput()->GetDirection());
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

  itk::ImageRegionConstIteratorWithIndex<InputMaskImageType> maskImageIterator(maskReader->GetOutput(), maskReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<OutputImageType> outputImageIterator(outputImage, outputImage->GetLargestPossibleRegion());

  double mean = 0;
  double distance = 0;
  unsigned int lowerQuartile = 0;
  unsigned int upperQuartile = 0;
  int size = 0;
  unsigned int i = 0;
  unsigned int counter = 0;
  MaskPixelType atlasValue;
  std::vector<DataPixelType> listOfDataValues;
  typename InputMaskImageType::IndexType maskIndex;
  typename InputMaskImageType::IndexType regionIndex;
  bool checkingAtlas = (args.atlasImage.length() > 0 ? true : false);

  for (maskImageIterator.GoToBegin(),
       outputImageIterator.GoToBegin();
       !maskImageIterator.IsAtEnd();
       ++outputImageIterator,
       ++maskImageIterator
       )
    {
      if (maskImageIterator.Get() != args.backgroundMask)
        {
    	    maskIndex = maskImageIterator.GetIndex();

          if (checkingAtlas)
            {
              atlasValue = atlasReader->GetOutput()->GetPixel(maskIndex);
            }

          for (i = 0; i < Dimension; i++)
            {
        	    regionIndex[i] = maskIndex[i] - ((windowSize[i]-1)/2);
            }

          InputImageRegionType region;
          region.SetSize(windowSize);
          region.SetIndex(regionIndex);

          itk::ImageRegionConstIteratorWithIndex<InputMaskImageType> maskRegionIterator(maskReader->GetOutput(), region);
          itk::ImageRegionConstIteratorWithIndex<InputDataImageType> dataRegionIterator(dataReader->GetOutput(), region);

          listOfDataValues.clear();

          for (maskRegionIterator.GoToBegin(),
        	     dataRegionIterator.GoToBegin();
               !maskRegionIterator.IsAtEnd();
               ++maskRegionIterator,
               ++dataRegionIterator)
            {
              distance = 0;
              regionIndex = maskRegionIterator.GetIndex();

              if (   (!checkingAtlas)
                  || (checkingAtlas && atlasReader->GetOutput()->GetPixel(regionIndex) != args.backgroundAtlas))
              {
                if (dataRegionIterator.Get() != args.backgroundData)
                {
                  if (  (!args.justMask)
                     || (args.justMask && maskRegionIterator.Get() != 0)
                     )
                    {
                      for (i = 0; i < Dimension; i++)
                        {
                          distance += (((regionIndex[i] - maskIndex[i]) * imageSpacing[i])
                                    *((regionIndex[i] - maskIndex[i]) * imageSpacing[i]));
                        }
                      distance = sqrt(distance);

                      if (distance <= args.radius)
                        {
                          listOfDataValues.push_back(dataRegionIterator.Get());
                        }
                    } // end mask check
                } // end data background check
              } // end atlas check
            } // end for each voxel in region

          // Now calculate Inter Quartile Mean.
          mean = 0;
          counter = 0;
          size = listOfDataValues.size();
          if (size > 0)
            {
              if (!args.doMean)
                {
                  sort(listOfDataValues.begin(), listOfDataValues.end());

                  lowerQuartile = (unsigned int)((size-1) * 0.25);
                  upperQuartile = (unsigned int)((size-1) * 0.75);

                }
              else
                {
                  lowerQuartile = 0;
                  upperQuartile = size - 1;
                }

              for (i = lowerQuartile; i <= upperQuartile; i++)
                {
                  mean += listOfDataValues[i];
                  counter ++;
                }
              if (counter > 0)
              {
                  mean /= (double)counter;

              }

            } // end size check
          outputImageIterator.Set(mean);
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
  args.justMask = false;
  args.backgroundData = 0;
  args.backgroundAtlas = 0;
  args.backgroundMask = 0;
  args.doMean = false;
  

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
    else if(strcmp(argv[i], "-m") == 0){
      args.maskImage=argv[++i];
      std::cout << "Set -m=" << args.maskImage << std::endl;
    }
    else if(strcmp(argv[i], "-atlas") == 0){
      args.atlasImage=argv[++i];
      std::cout << "Set -atlas=" << args.atlasImage << std::endl;
    }
    else if(strcmp(argv[i], "-radius") == 0){
      args.radius=atof(argv[++i]);
      std::cout << "Set -radius=" << niftk::ConvertToString(args.radius) << std::endl;
    }
    else if(strcmp(argv[i], "-justMask") == 0){
      args.justMask=true;
      std::cout << "Set -justMask=" << niftk::ConvertToString(args.justMask) << std::endl;
    }
    else if(strcmp(argv[i], "-bgData") == 0){
      args.backgroundData=atof(argv[++i]);
      std::cout << "Set -bgData=" << niftk::ConvertToString(args.backgroundData) << std::endl;
    }
    else if(strcmp(argv[i], "-bgAtlas") == 0){
      args.backgroundAtlas=atoi(argv[++i]);
      std::cout << "Set -bgAtlas=" << niftk::ConvertToString(args.backgroundAtlas) << std::endl;
    }
    else if(strcmp(argv[i], "-bgMask") == 0){
      args.backgroundMask=atoi(argv[++i]);
      std::cout << "Set -bgMask=" << niftk::ConvertToString(args.backgroundMask) << std::endl;
    }
    else if(strcmp(argv[i], "-mean") == 0){
      args.doMean=true;
      std::cout << "Set -mean=" << niftk::ConvertToString(args.doMean) << std::endl;
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
