/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-21 14:43:44 +0000 (Mon, 21 Nov 2011) $
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
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"

void Usage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Implements the subsampling described in section 3.2 and 3.2.1 of Acosta et. al. MIA 13 (2009) 730-743 doi:10.1016/j.media.2009.07.03" << std::endl;
  std::cout << "  This was used to generate low resolution test images from very high resolution ones." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << name << " -i <filename> -o <filename> [options] " << std::endl;
  std::cout << "  " << std::endl;  
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i    <filename>        Input image" << std::endl;
  std::cout << "    -o    <filename>        Output subsampled image" << std::endl << std::endl;      
  std::cout << "*** [options]   ***" << std::endl << std::endl;   
  std::cout << "    -sx   <int>   [2]       Subsampling factor" << std::endl;
  std::cout << "    -sy   <int>   [2]       Subsampling factor" << std::endl;
  std::cout << "    -sz   <int>   [2]       Subsampling factor" << std::endl;
}

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  int subsamplingFactorX;  
  int subsamplingFactorY;
  int subsamplingFactorZ;
  
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef float OutputPixelType;
  typedef typename itk::Image< PixelType, Dimension > ImageType;
  typedef typename itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef typename itk::ImageFileReader< ImageType  > ImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType > OutputWriterType;
  
  typename ImageReaderType::Pointer  imageReader  = ImageReaderType::New();
  imageReader->SetFileName(  args.inputImage );
  

  try 
    { 
      std::cout << "Loading image:" + args.inputImage << std::endl;
      imageReader->Update();
      std::cout << "Done" << std::endl;

    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  typedef typename ImageType::RegionType    RegionType;
  typedef typename ImageType::SizeType      SizeType;
  typedef typename ImageType::SpacingType   SpacingType;
  typedef typename ImageType::DirectionType DirectionType;
  typedef typename ImageType::PointType     OriginType;
  typedef typename ImageType::IndexType     IndexType;
  
  typename ImageType::Pointer inputImage = ImageType::New();
  inputImage = imageReader->GetOutput();
  
  SizeType inputSize = imageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
  IndexType inputIndex = imageReader->GetOutput()->GetLargestPossibleRegion().GetIndex();
  SpacingType inputSpacing = imageReader->GetOutput()->GetSpacing();
  OriginType inputOrigin = imageReader->GetOutput()->GetOrigin();
  DirectionType inputDirection = imageReader->GetOutput()->GetDirection();
  
  std::cout << "Input image size=" << inputSize \
    << ", spacing=" << inputSpacing \
    << ", origin=" << inputOrigin \
    << ", index=" << inputIndex \
    << ", direction=\n" << inputDirection \
    << std::endl;
  
  RegionType outputRegion;
  SizeType outputSize;
  IndexType outputIndex;
  SpacingType outputSpacing;
  DirectionType outputDirection;
  OriginType outputOrigin;
  
  if (Dimension == 2)
    {
      outputSize[0] = inputSize[0] / args.subsamplingFactorX;
      outputSize[1] = inputSize[1] / args.subsamplingFactorY;
    }
  else
    {
      outputSize[0] = inputSize[0] / args.subsamplingFactorX;
      outputSize[1] = inputSize[1] / args.subsamplingFactorY;      
      outputSize[2] = inputSize[2] / args.subsamplingFactorZ;
    }
  
  for (unsigned int i = 0; i < Dimension; i++)
    {
      outputIndex[i] = 0;
      outputSpacing[i] = (inputSpacing[i] * inputSize[i] / outputSize[i]); 
      outputOrigin[i] = inputOrigin[i] + ((inputSize[i]-1)*inputSpacing[i]/2.0)-((outputSize[i]-1)*outputSpacing[i]/2.0);
    }
  
  outputDirection = inputDirection;
  outputRegion.SetSize(outputSize);
  outputRegion.SetIndex(outputIndex);

  std::cout << "Output image size=" << outputSize \
    << ", spacing=" << outputSpacing \
    << ", origin=" << outputOrigin \
    << ", index=" << outputIndex \
    << ", direction=\n" << outputDirection \
    << std::endl;

  typename OutputImageType::Pointer outputImage = OutputImageType::New();
  outputImage->SetRegions(outputRegion);
  outputImage->SetSpacing(outputSpacing);
  outputImage->SetDirection(outputDirection);
  outputImage->SetOrigin(outputOrigin);
  outputImage->Allocate();
  outputImage->FillBuffer(0);
  
  unsigned long int counter;
  unsigned long int totalSize = 1;
  
  RegionType localRegion;
  SizeType localRegionSize;
  IndexType localRegionIndex;

  for (unsigned int i = 0; i < Dimension; i++)
    {
      localRegionSize[i] =  inputSize[i]/outputSize[i];
      localRegionIndex[i] = 0;
      totalSize *= localRegionSize[i]; 
    }

  localRegion.SetSize(localRegionSize);
  localRegion.SetIndex(localRegionIndex);

  std::cout << "Averaging over region size:" << localRegionSize << std::endl;
  
  typename itk::ImageRegionIteratorWithIndex<OutputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  for (outputIterator.GoToBegin();
       !outputIterator.IsAtEnd();
       ++outputIterator)
    {
      counter = 0;
      outputIndex = outputIterator.GetIndex();

      for (unsigned int i = 0; i < Dimension; i++)
        {
          localRegionIndex[i] = outputIndex[i] * localRegionSize[i];
        }
      localRegion.SetIndex(localRegionIndex);
      
      typename itk::ImageRegionConstIterator<ImageType> inputIterator(inputImage, localRegion);
      for (inputIterator.GoToBegin();
           !inputIterator.IsAtEnd();
           ++inputIterator)
        {
          if (inputIterator.Get() > 0)
            {
              counter++;
            }
        }
      
      outputIterator.Set(counter/(double)totalSize);
    }
  
  typename OutputWriterType::Pointer imageWriter = OutputWriterType::New();
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
 * \brief Implements the subsampling mentioned in section 3.2 and 3.2.1 of  Acosta et. al. MIA 13 (2009) 730-743 doi:10.1016/j.media.2009.07.03
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.subsamplingFactorX = 2;
  args.subsamplingFactorY = 2;
  args.subsamplingFactorZ = 2;
  

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
    else if(strcmp(argv[i], "-sx") == 0){
      args.subsamplingFactorX=atoi(argv[++i]);
      std::cout << "Set -sx=" << niftk::ConvertToString(args.subsamplingFactorX) << std::endl;
    }
    else if(strcmp(argv[i], "-sy") == 0){
      args.subsamplingFactorY=atoi(argv[++i]);
      std::cout << "Set -sy=" << niftk::ConvertToString(args.subsamplingFactorY) << std::endl;
    }
    else if(strcmp(argv[i], "-sz") == 0){
      args.subsamplingFactorZ=atoi(argv[++i]);
      std::cout << "Set -sz=" << niftk::ConvertToString(args.subsamplingFactorZ) << std::endl;
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
