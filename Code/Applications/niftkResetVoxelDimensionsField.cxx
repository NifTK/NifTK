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
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCommandLineHelper.h"
#include "itkImageRegionIterator.h"
#include "itkContinuousIndex.h"
#include "itkVector.h"

/*!
 * \file niftkResetVoxelDimensionsField.cxx
 * \page niftkResetVoxelDimensionsField
 * \section niftkResetVoxelDimensionsFieldSummary Loads an image in, and sets the voxel size to the ones you specified.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Loads an image in, and sets the voxel size to the ones you specified." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i       <filename>        Input image " << std::endl;
    std::cout << "    -o       <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl; 
    std::cout << "    -spacing x y z      [1]    Set the spacing of the output image" << std::endl;
  }

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  float spacing[3];
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  

  typedef typename itk::Image< PixelType, Dimension >     InputImageType;
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);

  try
  {
    imageReader->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  {
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }

  typename InputImageType::SizeType oldSize;
  typename InputImageType::SpacingType oldSpacing;
  typename InputImageType::PointType oldOrigin;
  typename InputImageType::DirectionType oldDirection;

  typename InputImageType::Pointer inputImage = imageReader->GetOutput();

  oldSize = inputImage->GetLargestPossibleRegion().GetSize();
  oldSpacing = inputImage->GetSpacing();
  oldOrigin = inputImage->GetOrigin();
  oldDirection = inputImage->GetDirection();

  typedef itk::ContinuousIndex<double, Dimension> ContinuousIndexType;
  typedef itk::Vector<double, Dimension> VectorType;

  typename InputImageType::IndexType cornerOfImageInVoxels;
  typename InputImageType::PointType cornerOfImageInMillimetres;
  typename InputImageType::PointType centreOfImageInMillimetres;

  ContinuousIndexType centreOfImageInVoxels;
  VectorType directionVector;

  for (int i = 0; i < Dimension; i++)
  {
    cornerOfImageInVoxels[i] = 0;
    centreOfImageInVoxels[i] = ((oldSize[i]-1)/2.0);
  }

  inputImage->TransformIndexToPhysicalPoint(cornerOfImageInVoxels, cornerOfImageInMillimetres);
  inputImage->TransformContinuousIndexToPhysicalPoint(centreOfImageInVoxels, centreOfImageInMillimetres);

  for (int i = 0; i < Dimension; i++)
  {
    directionVector[i] = centreOfImageInMillimetres[i] - cornerOfImageInMillimetres[i];
  }
  directionVector /= directionVector.GetNorm();

  typename InputImageType::SpacingType newSpacing;
  typename InputImageType::PointType newOrigin;

  for (int i = 0; i < Dimension; i++)
  {
    newSpacing[i] = args.spacing[i];
  }

  double diagonalLength = 0;
  for (int i = 0; i < Dimension; i++)
  {
    diagonalLength += newSpacing[i]*((oldSize[i]-1)/2.0)*newSpacing[i]*((oldSize[i]-1)/2.0);
  }  
  diagonalLength = sqrt(diagonalLength);

  for (int i = 0; i < Dimension; i++)
  {
    newOrigin[i] = centreOfImageInMillimetres[i] - diagonalLength*directionVector[i];
  }
  
  // Create new image, as a copy of the input

  typename InputImageType::Pointer image = InputImageType::New();
  image->SetRegions(inputImage->GetLargestPossibleRegion());
  image->SetDirection(inputImage->GetDirection());
  image->SetOrigin(newOrigin);
  image->SetSpacing(newSpacing);
  image->Allocate();
  image->FillBuffer(0);

  itk::ImageRegionIterator<InputImageType> inputIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<InputImageType> outputIterator(image, image->GetLargestPossibleRegion());
  for (inputIterator.GoToBegin(), outputIterator.GoToBegin();
       !inputIterator.IsAtEnd();
       ++inputIterator, 
       ++outputIterator)
  {
    outputIterator.Set(inputIterator.Get());
  }

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput(image);

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
 * \brief Takes image and changes voxel sizes in header.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
   
  // Set defaults
  for (unsigned int i = 0; i < 3; i++)
    {
      args.spacing[i] = 1;
    }

  

  // Parse command line args for image
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
  
  // Now we know we have 3 dimensions, parse the other command line args.
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-spacing") == 0){
      for (int j = 0; j < dims; j++)
        {
          args.spacing[j]=atof(argv[++i]);
          std::cout << "Set -spacing[" << niftk::ConvertToString(j) << "]=" << niftk::ConvertToString(args.spacing[j]) << std::endl;
        }
    }                
  }
  
  int result;

  // You could template for 2D and 3D, and all datatypes, but 64bit gcc compilers seem
  // to struggle here, so I've just done the bare minimum for now.

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
