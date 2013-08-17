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
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkCommandLineHelper.h>
#include <itkImageRegionIterator.h>
#include <itkConversionUtils.h>
#include <itkSpatialOrientationAdapter.h>

/*!
 * \file niftkResetDirectionField.cxx
 * \page niftkResetDirectionField
 * \section niftkResetDirectionFieldSummary Takes a copy of an input image, copying data into another image, where the origin, spacing and direction can be over-written.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "Takes a copy of an input image, copying data into another image, where the origin, spacing and direction can be over-written." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i       <filename>        Input image " << std::endl;
    std::cout << "    -o       <filename>        Output image" << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl; 
    std::cout << "    -default <float>             [0]    Default pixel value " << std::endl;
    std::cout << "    -origin  x y z               [0]    Set the origin of the output image" << std::endl;
    std::cout << "    -spacing x y z               [1]    Set the spacing of the output image" << std::endl;
    std::cout << "    -direction a b c d e f g h i        Set the direction matrix" << std::endl;
    std::cout << "    -orientation <string>               Three letter orientation code, upper case, made from valid combinations of L/R, I/S and A/P" << std::endl;
    std::cout << "                                        This will compute a set of direction cosines, so can be used to set the cosines, without flipping data" << std::endl;
  }

struct arguments
{
  std::string inputImage;
  std::string outputImage;
  std::string orientation;
  float defaultPixelValue;
  float origin[3];
  float spacing[3];
  float direction[9];
  bool directionSetByUser;
  bool originSetByUser;
  bool spacingSetByUser;
};

template <int Dimension>
void SetDirectionToIdentity(arguments& args)
{
  for (unsigned int i = 0; i < Dimension; i++)
  {
    args.direction[i*Dimension + i] = 1;
  }
}

template <int Dimension, class PixelType> 
int DoMain(arguments& args)
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

  if (!args.directionSetByUser)
  {
    SetDirectionToIdentity<Dimension>(args);
  }

  typename InputImageType::RegionType regionType;
  typename InputImageType::SpacingType spacingType;
  typename InputImageType::PointType originType;
  typename InputImageType::DirectionType directionType;
  
  for (unsigned int i = 0; i < Dimension; i++)
    {
      spacingType[i] = args.spacing[i];
      originType[i] = args.origin[i];
    }

  for (unsigned int y = 0; y < Dimension; y++)
  {
    for (unsigned int x = 0; x < Dimension; x++)
    {
      int counter=y*Dimension + x;
      directionType[y][x] = args.direction[counter];
    }
  }

  regionType = imageReader->GetOutput()->GetLargestPossibleRegion();

  // Create new image
  typename InputImageType::Pointer outputImage = InputImageType::New();
  outputImage->SetRegions(regionType);
  if (args.spacingSetByUser)
  {
    outputImage->SetSpacing(spacingType);
  }
  else
  {
    outputImage->SetSpacing(imageReader->GetOutput()->GetSpacing());
  }
  if (args.originSetByUser)
  {
    outputImage->SetOrigin(originType);
  }
  else
  {
    outputImage->SetOrigin(imageReader->GetOutput()->GetOrigin());
  }
  if (args.directionSetByUser)
  {
    outputImage->SetDirection(directionType);
  }
  else
  {
    outputImage->SetDirection(imageReader->GetOutput()->GetDirection());
  }
  if (args.orientation.length() != 0)
  {
    typename itk::SpatialOrientation::ValidCoordinateOrientationFlags orientation = itk::ConvertStringToSpatialOrientation(args.orientation);
    typedef itk::SpatialOrientationAdapter AdaptorType;
    AdaptorType adaptor;
    typename InputImageType::DirectionType dir = adaptor.ToDirectionCosines(orientation);
    outputImage->SetDirection(dir);
  }

  outputImage->Allocate();
  outputImage->FillBuffer((PixelType)args.defaultPixelValue);

  itk::ImageRegionIterator<InputImageType> inputIterator(imageReader->GetOutput(), imageReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<InputImageType> outputIterator(outputImage, outputImage->GetLargestPossibleRegion());
  for (inputIterator.GoToBegin(), outputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator, ++outputIterator)
  {
    outputIterator.Set(inputIterator.Get());
  }

  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput(outputImage);
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
 * \brief Takes image and does binary thresholding in ITK style.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
   
  // Set defaults
  args.defaultPixelValue = 0;
  args.directionSetByUser = false;
  args.spacingSetByUser = false;
  args.originSetByUser = false;

  for (unsigned int i = 0; i < 9; i++)
    {
      args.direction[i] = 0;
    }

  for (unsigned int i = 0; i < 3; i++)
    {
      args.origin[i] = 0;
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
    else if(strcmp(argv[i], "-orientation") == 0){
      args.orientation=argv[++i];
      std::cout << "Set -orientation=" << args.orientation << std::endl;
    }
    else if(strcmp(argv[i], "-default") == 0){
      args.defaultPixelValue=atof(argv[++i]);
      std::cout << "Set -default=" << niftk::ConvertToString(args.defaultPixelValue) << std::endl;
    }    
  }

  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  int dims = itk::PeekAtImageDimension(args.inputImage);
  if (dims != 3)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }
  
  // Now we know we have 3 dimensions, parse the other command line args.
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-spacing") == 0){
      args.spacingSetByUser = true;
      for (int j = 0; j < dims; j++)
        {
          args.spacing[j]=atof(argv[++i]);
          std::cout << "Set -spacing[" << niftk::ConvertToString(j) << "]=" << niftk::ConvertToString(args.spacing[j]) << std::endl;
        }
    }            
    else if(strcmp(argv[i], "-origin") == 0){
      args.originSetByUser = true;
      for (int j = 0; j < dims; j++)
        {
          args.origin[j]=atof(argv[++i]);
          std::cout << "Set -origin[" << niftk::ConvertToString(j) << "]=" << niftk::ConvertToString(args.origin[j]) << std::endl;
        }
    }
    else if(strcmp(argv[i], "-direction") == 0){
      args.directionSetByUser = true;
      for (int j = 0; j < dims*dims; j++)
        {
          args.direction[j]=atof(argv[++i]);
          std::cout << "Set -direction[" << niftk::ConvertToString(j) << "]=" << niftk::ConvertToString(args.direction[j]) << std::endl;
        }
    }

  }
  
  int result;

  switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::UCHAR:
      result = DoMain<3, unsigned char>(args);
      break;
    case itk::ImageIOBase::CHAR:
      result = DoMain<3, char>(args);
      break;
    case itk::ImageIOBase::USHORT:
      result = DoMain<3, unsigned short>(args);
      break;
    case itk::ImageIOBase::SHORT:
      result = DoMain<3, short>(args);
      break;
    case itk::ImageIOBase::UINT:
      result = DoMain<3, unsigned int>(args);
      break;
    case itk::ImageIOBase::INT:
      result = DoMain<3, int>(args);
      break;
    case itk::ImageIOBase::ULONG:
      result = DoMain<3, unsigned long>(args);
      break;
    case itk::ImageIOBase::LONG:
      result = DoMain<3, long>(args);
      break;
    case itk::ImageIOBase::FLOAT:
      result = DoMain<3, float>(args);
      break;
    case itk::ImageIOBase::DOUBLE:
      result = DoMain<3, double>(args);
      break;
    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  return result;
}
