/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>
#include "itkCommandLineHelper.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNifTKImageIOFactory.h>
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

struct niftk::CommandLineArgumentDescription clArgList[] = 
{
  {OPT_STRING|OPT_REQ, "i", "filename", "Input image."},
  {OPT_STRING|OPT_REQ, "o", "filename", "Output image."},

  {OPT_STRING, "l", "string", "Comma separated list of voxel intensities (labels in an atlas image for example)."},
  {OPT_FLOAT, "bg", "float", "Output background value. Default 0."},
  {OPT_FLOAT, "fg", "float", "Output foreground value. Default 1."},

  {OPT_DONE, NULL, NULL, 
   "Given an image and a list of labels, will output a binary mask, where all voxels with"
   "voxel intensities matching the list are output as foreground, and all other voxels are background.\n"
  }
};

enum
{
  O_INPUT_IMAGE, 

  O_OUTPUT_IMAGE, 

  O_LABELS,

  O_BACKGROUND_VALUE,

  O_FOREGROUND_VALUE
};

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
  itk::NifTKImageIOFactory::Initialize();

  // To pass around command line args
  struct arguments args;
  args.backgroundValue = 0;
  args.forgroundValue = 1;

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, false);

  CommandLineOptions.GetArgument( O_INPUT_IMAGE, args.inputImage );

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, args.outputImage );
  
  CommandLineOptions.GetArgument( O_LABELS, args.regionNumbers );
  
  CommandLineOptions.GetArgument( O_BACKGROUND_VALUE, args.backgroundValue );
  
  CommandLineOptions.GetArgument( O_FOREGROUND_VALUE, args.forgroundValue );
  
  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 2 && dims != 3)
    {
      std::cout << "Unsupported image dimension" << std::endl;
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
