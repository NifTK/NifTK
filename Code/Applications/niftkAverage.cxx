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
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

/*!
 * \file niftkAverage.cxx
 * \page niftkAverage
 * \section niftkAverageSummary Uses ITK ImageFileReader to load any number of input images, creating the arithmetic mean on a voxel by voxel basis, writing the output with ITK ImageFileWriter.
 *
 * This program uses ITK ImageFileReaders to load any number of images, and then uses simple ITK iterators to accumulate
 * the data into a single image before calculating the arithmetic mean. The output is written using ITK ImageFileWriter.
 *
 * \li Dimensions: 2,3
 * \li Pixel type: All input images are converted to float on input.
 *
 * \section niftkAverageCaveat Caveats
 * \li All images must have the same size, determined by an ITK Region, which checks the Region Size and Index.
 */

void Usage(char *exec)
  {
	niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Uses ITK ImageFileReader to load any number of input images, creating the arithmetic mean on a voxel by voxel basis, writing the output with ITK ImageFileWriter. All input images must be the same size, and are converted to float on input, and hence are float on output." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -o outputImage -i inputImage " << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>        Input image (repeated) " << std::endl;
    std::cout << "    -o    <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
  }

struct arguments
{
  std::vector<std::string> inputImages;
  std::string outputImage;  
};

template <int Dimension> 
int DoMain(arguments args)
{  
  
  typedef float PixelType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileReader<ImageType>  ImageFileReaderType;
  typedef itk::ImageFileWriter<ImageType>  ImageFileWriterType;
  
  try
  {
    int counter = 1;
    
    typename ImageFileReaderType::Pointer fileReader = ImageFileReaderType::New();
    fileReader->SetFileName(args.inputImages[0]);
    fileReader->Update();
    
    typename ImageType::Pointer image = fileReader->GetOutput();
    image->DisconnectPipeline();
    
    for (unsigned int i = 0; i < args.inputImages.size(); i++)
      {
        fileReader->SetFileName(args.inputImages[i]);
        fileReader->Update();
        
        if (image->GetLargestPossibleRegion().GetSize() != fileReader->GetOutput()->GetLargestPossibleRegion().GetSize())
          {
            std::cerr << "The " << i+1 << "th image has a different size to the first image" << std::endl;
            return EXIT_FAILURE;
          }
        
        typename itk::ImageRegionConstIterator<ImageType> inputIterator(fileReader->GetOutput(), fileReader->GetOutput()->GetLargestPossibleRegion());
        typename itk::ImageRegionIterator<ImageType> outputIterator(image, image->GetLargestPossibleRegion());
        
        for (inputIterator.GoToBegin(), 
            outputIterator.GoToBegin();
             !inputIterator.IsAtEnd();
             ++inputIterator,
             ++outputIterator)
          {
            outputIterator.Set(inputIterator.Get() + outputIterator.Get());
          }
        
        counter++;
        
        std::cout << "Averaged " << args.inputImages[i] << std::endl;
      }
    
    typename itk::ImageRegionIterator<ImageType> outputIterator(image, image->GetLargestPossibleRegion());
    for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
      {
        outputIterator.Set(outputIterator.Get() / (float)counter);
      }
    
    typename ImageFileWriterType::Pointer fileWriter = ImageFileWriterType::New();
    fileWriter->SetFileName(args.outputImage);
    fileWriter->SetInput(image);
    fileWriter->Update();
    
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }  
  return 0;
}

/**
 * \brief Takes image1 and image2 and adds them together
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputImage=argv[++i];
      std::cout << "Set -o=" << args.outputImage<< std::endl;
    }
    else if(strcmp(argv[i], "-i") == 0){
      std::string tmp = std::string(argv[++i]);
      args.inputImages.push_back(tmp);
      std::cout << "Set -i=" << tmp<< std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if (args.inputImages.size() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  for (unsigned int i = 0; i < args.inputImages.size(); i++)
    {
      std::string tmp = args.inputImages[0];
      if (tmp.length() == 0)
        {
          std::cerr << "The " << i+1 << "th string has zero length" << std::endl;
          return EXIT_FAILURE;      
        }
    }
  
  int dims = itk::PeekAtImageDimension(args.inputImages[0]);
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
