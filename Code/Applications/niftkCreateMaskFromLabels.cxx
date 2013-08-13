/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <set>
#include <algorithm>

/*!
 * \file niftkCreateMaskFromLabels.cxx
 * \page niftkCreateMaskFromLabels
 * \section niftkCreateMaskFromLabelsSummary Given an image and a list of labels, will output a binary mask, where all voxels with voxel intensities matching the list are output as foreground, and all other voxels are background.
 *
 * This program uses ITK ImageFileReaders to load an image, creates an output image programmatically and
 * then simply compares voxels against the input list of intensities. If the voxel matches the list, the output is
 * a foreground pixel and if the voxels does not match, the output is a background voxel.
 *
 * \li Dimensions: 2,3
 * \li Pixel type: Scalars only, of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float, double
 *
 * \section niftkCreateMaskFromLabelsCaveat Caveats
 */

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Given an image and a list of labels, will output a binary mask, where all voxels with" << std::endl;
    std::cout << "  voxel intensities matching the list are output as foreground, and all other voxels are background." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
    std::cout << "    -l 1,2,3,4,5            Comma separated list of voxel intensities (labels in an atlas image for example)." << std::endl;    
    std::cout << "    -bg <float>             Output background value. Default 0." << std::endl;
    std::cout << "    -fg <float>             Output foreground value. Default 1." << std::endl;
  }

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  std::string regionNumbers;
  float backgroundValue;
  float forgroundValue;
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;
  
  try
  {
    
    typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
    imageReader->SetFileName(args.inputImage);
    

    std::cout << "Loading image:" << args.inputImage << std::endl;
    imageReader->SetFileName(args.inputImage);
    imageReader->Update();
    std::cout << "Done" << std::endl;
 
    std::set<PixelType> set;
    char * tok = strtok(const_cast<char*>(args.regionNumbers.c_str()), ",");
    while (tok != NULL) {
      PixelType value = (PixelType)(atof(tok));
      set.insert(value);
      tok = strtok(NULL,",");
    }

    typename InputImageType::Pointer outputImage = InputImageType::New();
    outputImage->SetRegions(imageReader->GetOutput()->GetLargestPossibleRegion());
    outputImage->SetSpacing(imageReader->GetOutput()->GetSpacing());
    outputImage->SetDirection(imageReader->GetOutput()->GetDirection());
    outputImage->SetOrigin(imageReader->GetOutput()->GetOrigin());
    outputImage->Allocate();
    outputImage->FillBuffer(0);
    
    itk::ImageRegionConstIterator<InputImageType> inputIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<InputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
    PixelType inputValue;
    
    for (inputIterator.GoToBegin(),
         outputIterator.GoToBegin();
         !inputIterator.IsAtEnd();
         ++inputIterator,
         ++outputIterator)
      {
        inputValue = inputIterator.Get();
        if (!(set.find(inputValue) == set.end()))
          {
            outputIterator.Set((PixelType)(args.forgroundValue));
          }
        else
          {
            outputIterator.Set((PixelType)(args.backgroundValue));
          }
      }
    
    std::cout << "Saving image:" << args.outputImage << std::endl;
    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
    imageWriter->SetFileName(args.outputImage);
    imageWriter->SetInput(outputImage);
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
 * \brief An image, and a list of 
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.backgroundValue = 0;
  args.forgroundValue = 1;
  

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
    else if(strcmp(argv[i], "-bg") == 0){
      args.backgroundValue=atof(argv[++i]);
      std::cout << "Set -bg=" << niftk::ConvertToString(args.backgroundValue) << std::endl;
    }
    else if(strcmp(argv[i], "-fg") == 0){
      args.forgroundValue=atof(argv[++i]);
      std::cout << "Set -fg=" << niftk::ConvertToString(args.forgroundValue) << std::endl;
    }
    else if(strcmp(argv[i], "-l") == 0){
      args.regionNumbers=argv[++i];
      std::cout << "Set -l=" << args.regionNumbers << std::endl;
    }        
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0)
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
          result = DoMain<2, float>(args);  
        }
      else
        {
          result = DoMain<3, float>(args);
        }
      break;
    case itk::ImageIOBase::DOUBLE:
      if (dims == 2)
        {
          result = DoMain<2, double>(args);  
        }
      else
        {
          result = DoMain<3, double>(args);
        }
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
