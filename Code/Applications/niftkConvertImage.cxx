/*=============================================================================

  NifTK: An image processing toolkit jointly developed by the
  Dementia Research Centre, and the Centre For Medical Image Computing
  at University College London.

  See:        http://dementia.ion.ucl.ac.uk/
  http://cmic.cs.ucl.ac.uk/
  http://www.ucl.ac.uk/

  Last Changed      : $Date: 2011-09-20 20:35:56 +0100 (Tue, 20 Sep 2011) $
  Revision          : $Revision: 7340 $
  Last modified by  : $Author: ad $

  Original author   : m.clarkson@ucl.ac.uk
  m.modat@ucl.ac.uk

  Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

  ============================================================================*/

#include <map>

#include "itkLogHelper.h"
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageIOFactory.h"
#include "itkImageIOBase.h"
#include "itkGiplImageIO.h"
#include "itkVTKImageIO.h"
#include "itkINRImageIOFactory.h"
#include "itkVectorImage.h"
#include "itkRGBPixel.h"

#include "niftkConvertImageCLP.h"

/*!
 * \file niftkConvertImage.cxx
 * \page niftkConvertImage
 * \section niftkConvertImageSummary Converts an input file to an output file. Originally based on Marc Modat's convertImage.
 *
 * \li Dimensions: 2,3
 *
 * \section niftkConvertImageCaveat Caveats
 */

void Usage(char *exec)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl;
  std::cout << "  Converts an input file to an output file. Originally based on Marc Modat's convertImage." << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
  std::cout << "  " << std::endl;
  std::cout << "*** [mandatory] ***" << std::endl << std::endl;
  std::cout << "    -i <filename>                 Input file " << std::endl;
  std::cout << "    -o <filename>                 Output file " << std::endl << std::endl;
  std::cout << "*** [options]   ***" << std::endl << std::endl;
  std::cout << "     -v                                Try to force 4D scalar data to 3D vectors" << std::endl;
  std::cout << "     -ot <uchar|ushort|char|short|int|float|double> Force the output type" << std::endl;
  return;
}

typedef struct arguments
{
  bool flg4DScalarImageTo3DVectorImage;

  float rx;
  float ry;
  float rz;
  float rt;

  std::string strOutputType;

  std::string fileInputImage;
  std::string fileOutputImage;

  arguments() {
    flg4DScalarImageTo3DVectorImage = false;

    rx = 0.;
    ry = 0.;
    rz = 0.;
    rt = 0.;
  }

} Arguments;

template <class TPixel> bool WriteNewImage(Arguments, const int dimension, bool vector);

template <class TPixel, const int dimension> bool WriteNewScalarImage(Arguments);

template <class TPixel, const int inputDimension, const int outputDimension> bool Write4DScalarImageTo3DVectorImage(Arguments);


// -------------------------------------------------------------------------------------
// main(int argc, char **argv)
// -------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
  // Map to associate the input strings with the enum values
  static std::map<std::string, itk::ImageIOBase::IOComponentType> mapOutputTypes;

  mapOutputTypes["unset"]  = itk::ImageIOBase::UNKNOWNCOMPONENTTYPE;
  mapOutputTypes["uchar"]  = itk::ImageIOBase::UCHAR;
  mapOutputTypes["char"]   = itk::ImageIOBase::CHAR;
  mapOutputTypes["ushort"] = itk::ImageIOBase::USHORT;
  mapOutputTypes["short"]  = itk::ImageIOBase::SHORT;
  mapOutputTypes["uint"]   = itk::ImageIOBase::UINT;
  mapOutputTypes["int"]    = itk::ImageIOBase::INT;
  mapOutputTypes["ulong"]  = itk::ImageIOBase::ULONG;
  mapOutputTypes["long"]   = itk::ImageIOBase::LONG;
  mapOutputTypes["float"]  = itk::ImageIOBase::FLOAT;
  mapOutputTypes["double"] = itk::ImageIOBase::DOUBLE;
  
  mapOutputTypes["unsigned_char"]  = itk::ImageIOBase::UCHAR;
  mapOutputTypes["unsigned_short"] = itk::ImageIOBase::USHORT;
  mapOutputTypes["unsigned_int"]   = itk::ImageIOBase::UINT;
  mapOutputTypes["unsigned_long"]  = itk::ImageIOBase::ULONG;

  Arguments args;

  

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputImage.length() == 0 || fileOutputImage.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  args.flg4DScalarImageTo3DVectorImage = flg4DScalarImageTo3DVectorImage;

  args.rx = rx;
  args.ry = ry;
  args.rz = rz;
  args.rt = rt;

  args.fileInputImage  = fileInputImage;
  args.fileOutputImage = fileOutputImage;
  
  args.strOutputType = strOutputType;


  try
  {
      
    // Data variable type
    itk::ObjectFactoryBase::RegisterFactory(itk::INRImageIOFactory::New());

    std::cout << "Input             :\t" << args.fileInputImage << std::endl;
    std::cout << "Output            :\t" << args.fileOutputImage << std::endl;

    itk::ImageIOBase::Pointer imageIO = 
      itk::ImageIOFactory::CreateImageIO(args.fileInputImage.c_str(), itk::ImageIOFactory::ReadMode);

    imageIO->SetFileName(args.fileInputImage);
    imageIO->ReadImageInformation();

    int dimension=imageIO->GetNumberOfDimensions();
    int nNonUnityDimensions = dimension;
    itk::ImageIOBase::IOPixelType PixelType=imageIO->GetPixelType();

    if (strOutputType == std::string("unset"))
    {
      strOutputType = imageIO->GetComponentTypeAsString( imageIO->GetComponentType() ); 
    }
  

    for ( int i=0; i<dimension; i++ )  
    {
      if ( imageIO->GetDimensions( i ) <= 1 )
      {
	nNonUnityDimensions--;
      }
    }
    dimension = nNonUnityDimensions;

    std::cout << "Image Dimension   :\t" << dimension <<std::endl;
    std::cout << "PixelType         :\t" << imageIO->GetPixelTypeAsString(PixelType) << std::endl;
    std::cout << "Vector type       :\t" << args.flg4DScalarImageTo3DVectorImage << std::endl;
    std::cout << "Image voxel type  :\t";
    
    switch ( PixelType )
    {
    case itk::ImageIOBase::SCALAR:
    {

      switch ( mapOutputTypes[ strOutputType.c_str() ] )
      {
      case itk::ImageIOBase::UCHAR:
	std::cout<<"unsigned char"<<std::endl;
	WriteNewImage<unsigned char>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::CHAR:
	std::cout<<"char"<<std::endl;
	WriteNewImage<char>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::USHORT:
	std::cout<<"unsigned short"<<std::endl;
	WriteNewImage<unsigned short>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::SHORT:
	std::cout<<"short"<<std::endl;
	WriteNewImage<short>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::UINT:
	std::cout<<"unsigned int"<<std::endl;
	WriteNewImage<unsigned int>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::INT:
	std::cout<<"int"<<std::endl;
	WriteNewImage<int>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::ULONG:
	std::cout<<"unsigned long"<<std::endl;
	WriteNewImage<unsigned long>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::LONG:
	std::cout<<"long"<<std::endl;
	WriteNewImage<long>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::FLOAT:
	std::cout<<"float"<<std::endl;
	WriteNewImage<float>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::DOUBLE:
	std::cout<<"double"<<std::endl;
	WriteNewImage<double>(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      default:
	std::cerr << "non standard component type" << std::endl;
	return EXIT_FAILURE;
      }

      break;
    }

    case itk::ImageIOBase::RGB:
    {

      switch ( mapOutputTypes[ strOutputType.c_str() ] )
      {
      case itk::ImageIOBase::UCHAR:
	std::cout<<"unsigned char"<<std::endl;
	WriteNewImage< itk::RGBPixel<unsigned char> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::CHAR:
	std::cout<<"char"<<std::endl;
	WriteNewImage< itk::RGBPixel<char> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::USHORT:
	std::cout<<"unsigned short"<<std::endl;
	WriteNewImage< itk::RGBPixel<unsigned short> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::SHORT:
	std::cout<<"short"<<std::endl;
	WriteNewImage< itk::RGBPixel<short> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::UINT:
	std::cout<<"unsigned int"<<std::endl;
	WriteNewImage< itk::RGBPixel<unsigned int> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::INT:
	std::cout<<"int"<<std::endl;
	WriteNewImage< itk::RGBPixel<int> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::ULONG:
	std::cout<<"unsigned long"<<std::endl;
	WriteNewImage< itk::RGBPixel<unsigned long> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::LONG:
	std::cout<<"long"<<std::endl;
	WriteNewImage< itk::RGBPixel<long> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::FLOAT:
	std::cout<<"float"<<std::endl;
	WriteNewImage< itk::RGBPixel<float> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      case itk::ImageIOBase::DOUBLE:
	std::cout<<"double"<<std::endl;
	WriteNewImage< itk::RGBPixel<double> >(args, dimension, args.flg4DScalarImageTo3DVectorImage);
	break;
      default:
	std::cerr << "non standard component type" << std::endl;
	return EXIT_FAILURE;
      }

      break;
    }

    default:
      std::cerr << "non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cout << "ExceptionObject caught !" << std::endl; 
    std::cout << err << std::endl; 
    return EXIT_FAILURE;
  } 

  return EXIT_SUCCESS;
}


// -------------------------------------------------------------------------------------
// WriteNewImage(Arguments args, const int dimension, bool vec)
// -------------------------------------------------------------------------------------

template <class TPixel> bool WriteNewImage(Arguments args, const int dimension, bool vec)
{

  switch (dimension)
  {
  case 2:
    if (vec)
    {
      std::cerr << "Ignoring '-v' command line argument" << std::endl;
    }
    else
    {
      WriteNewScalarImage<TPixel, 2>(args);            
    }
    break;
  case 3:
    if (vec)
    {
      std::cerr << "Ignoring '-v' command line argument" << std::endl;
    }
    else
    {
      WriteNewScalarImage<TPixel, 3>(args);            
    }
    break;
  case 4:
    if (vec)
    {
      Write4DScalarImageTo3DVectorImage<TPixel, 4, 3>(args);    
    }
    else
    {
      WriteNewScalarImage<TPixel, 4>(args);            
    }
    break;
  case 5:
    if (vec)
    {
      std::cerr << "Ignoring '-v' command line argument" << std::endl;
    }
    else
    {
      WriteNewScalarImage<TPixel, 5>(args);            
    }
    break;
  }

  return EXIT_SUCCESS;
}


// -------------------------------------------------------------------------------------
// WriteNewScalarImage(Arguments args)
// -------------------------------------------------------------------------------------

template <class TPixel, const int dimension> bool WriteNewScalarImage(Arguments args)
{
  typedef itk::Image<TPixel, dimension>  ImageType;

  itk::ObjectFactoryBase::RegisterFactory(itk::INRImageIOFactory::New());

  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typename ImageReaderType::Pointer reader = ImageReaderType::New();

  // Image format
  const char *inputFileExt=strrchr(args.fileInputImage.c_str(), '.');
  if (strcmp(inputFileExt, ".gipl")==0)
  {
    itk::GiplImageIO::Pointer giplImageIO = itk::GiplImageIO::New();
    reader->SetImageIO(giplImageIO);
  }
  if (strcmp(inputFileExt, ".vtk")==0)
  {
    itk::VTKImageIO::Pointer vtkImageIO = itk::VTKImageIO::New();
    reader->SetImageIO(vtkImageIO);
  }

  reader->SetFileName(args.fileInputImage);
  try
  {
    reader->Update();
  }
  catch(itk::ExceptionObject &err)
  {
    std::cerr<<"Exception caught when reading the input image: "<< args.fileInputImage <<std::endl;
    std::cerr<<"Error: "<<err<<std::endl;
    return EXIT_FAILURE;
  }

  // Change the resolution?

  typename ImageType::Pointer image = reader->GetOutput();
  image->DisconnectPipeline();

  typename ImageType::SpacingType spacing = image->GetSpacing();

  std::cout << "Initial spacing: " << spacing << std::endl;

  if ( args.rx && ( dimension > 0 ) ) 
  {
    std::cout << "Modifying 'x' resolution from: " 
	      << spacing[0] << " to " << args.rx << " mm" << std::endl;
    spacing[0] = args.rx;
  }

  if ( args.ry && ( dimension > 1 ) ) 
  {
    std::cout << "Modifying 'y' resolution from: " 
	      << spacing[1] << " to " << args.ry << " mm" << std::endl;
    spacing[1] = args.ry;
  }

  if ( args.rz && ( dimension > 2 ) ) 
  {
    std::cout << "Modifying 'z' resolution from: " 
	      << spacing[2] << " to " << args.rz << " mm" << std::endl;
    spacing[2] = args.rz;
  }

  if ( args.rt && ( dimension > 3 ) ) {
    std::cout << "Modifying temporal resolution from: " 
	      << spacing[3] << " to " << args.rt << " s" << std::endl;
    spacing[3] = args.rt;
  }

  image->SetSpacing( spacing );

  std::cout << "New spacing: " << image->GetSpacing() << std::endl;

  image->Print(std::cout);

  typedef itk::ImageFileWriter<ImageType> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(image);
  writer->SetFileName(args.fileOutputImage);
  try
  {
    writer->Update();
  }
  catch(itk::ExceptionObject &err)
  {
    std::cerr<<"Exception caught when writing the output image: "<< args.fileOutputImage <<std::endl;
    std::cerr<<"Error: "<<err<<std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


// -------------------------------------------------------------------------------------
// Write4DScalarImageTo3DVectorImage(Arguments args)
// -------------------------------------------------------------------------------------

template <class TPixel, const int inputDimension, const int outputDimension> bool Write4DScalarImageTo3DVectorImage(Arguments args)
{
  typedef itk::Image<TPixel, inputDimension>  InputImageType;
  typedef itk::VectorImage<TPixel, outputDimension> OutputImageType;
    
  itk::ObjectFactoryBase::RegisterFactory(itk::INRImageIOFactory::New());

  typedef itk::ImageFileReader<InputImageType> ImageReaderType;
  typename ImageReaderType::Pointer reader = ImageReaderType::New();

  // Image format
  const char *inputFileExt=strrchr(args.fileInputImage.c_str(), '.');
  if (strcmp(inputFileExt, ".gipl")==0)
  {
    itk::GiplImageIO::Pointer giplImageIO = itk::GiplImageIO::New();
    reader->SetImageIO(giplImageIO);
  }
  if (strcmp(inputFileExt, ".vtk")==0)
  {
    itk::VTKImageIO::Pointer vtkImageIO = itk::VTKImageIO::New();
    reader->SetImageIO(vtkImageIO);
  }

  reader->SetFileName(args.fileInputImage);
  try
  {
    reader->Update();
  }
  catch(itk::ExceptionObject &err)
  {
    std::cerr<<"Exception caught when reading the input image: "<< args.fileInputImage <<std::endl;
    std::cerr<<"Error: "<<err<<std::endl;
    return EXIT_FAILURE;
  }

  typename OutputImageType::Pointer outputImage = OutputImageType::New();
  typename InputImageType::SizeType inputSize;
  typename InputImageType::SpacingType inputSpacing;
  typename InputImageType::IndexType inputIndex;
  typename InputImageType::PointType inputOrigin;
    
  typename OutputImageType::SizeType outputSize;
  typename OutputImageType::SpacingType outputSpacing;
  typename OutputImageType::IndexType outputIndex;
  typename OutputImageType::PointType outputOrigin;
  typename OutputImageType::RegionType outputRegion;
  typename OutputImageType::PixelType outputPixel;
    
  int outputPixelSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize()[inputDimension-1];
  inputSize = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
  inputSpacing = reader->GetOutput()->GetSpacing();
  inputOrigin = reader->GetOutput()->GetOrigin();
  inputIndex = reader->GetOutput()->GetLargestPossibleRegion().GetIndex();
    
  outputSize.Fill(1);
  outputSpacing.Fill(1);
  outputIndex.Fill(0);
  outputOrigin.Fill(0);
    
  for (unsigned int i = 0; i < outputDimension; i++)
  {
    outputSize[i] = inputSize[i];
    outputSpacing[i] = inputSpacing[i];
    outputOrigin[i] = inputOrigin[i];
    outputIndex[i] = inputIndex[i];
  }
    
  outputPixel.SetSize(outputPixelSize);
  outputImage->SetNumberOfComponentsPerPixel(outputPixelSize);

  outputRegion.SetSize(outputSize);
  outputRegion.SetIndex(outputIndex);

  outputImage->SetRegions(outputRegion);
  outputImage->SetSpacing(outputSpacing);
  outputImage->SetOrigin(outputOrigin);
    
  outputImage->Allocate();
    
  for(unsigned int x = 0; x < outputSize[0]; x++) 
  {
    for (unsigned int y = 0; y < outputSize[1]; y++)
    {
      for (unsigned int z = 0; z < outputSize[2]; z++)
      {
	for (unsigned int t = 0; t < (unsigned int)outputPixelSize; t++)
	{
	  inputIndex[0] = x;
	  inputIndex[1] = y;
	  inputIndex[2] = z;
	  inputIndex[3] = t;
                    
	  outputPixel[t] = reader->GetOutput()->GetPixel(inputIndex); 
	}
	outputIndex[0] = x;
	outputIndex[1] = y;
	outputIndex[2] = z;
	outputImage->SetPixel(outputIndex, outputPixel);
      }
    }
  }
    
    
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(outputImage);
  writer->SetFileName(args.fileOutputImage);
  try
  {
    writer->Update();
  }
  catch(itk::ExceptionObject &err)
  {
    std::cerr<<"Exception caught when writing the output image: "<< args.fileOutputImage <<std::endl;
    std::cerr<<"Error: "<<err<<std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

/* ********************************************************************** */
