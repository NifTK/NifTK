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
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkGDCMImageIO.h>


#include <niftkConvertImageToDICOMCLP.h>

/*!
 * \file niftkConvertImageToDICOM.cxx
 * \page niftkConvertImageToDICOM
 * \section niftkConvertImageToDICOMSummary Convert an image to DICOM format.
 *
 * This program converts an image to DICOM format by adding a DICOM header from an external source (e.g. an existing DICOM image from which the image was derived).
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkConvertImageToDICOMCaveats Caveats
 * \li None
 */
struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  std::string fileInputImage;
  std::string fileInputDICOM;
  std::string fileOutputImage;  
};

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;

  typedef itk::MetaDataDictionary DictionaryType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  typedef itk::GDCMImageIO           ImageIOType;

 
  // Read the image

  typename InputImageReaderType::Pointer imReader = InputImageReaderType::New();

  imReader->SetFileName( args.fileInputImage );

  try
  {
    std::cout << std::endl << "Reading image from file: " << args.fileInputImage << std::endl;
    imReader->Update(); 
  }
  catch( itk::ExceptionObject &err ) 
  { 
    std::cerr << "ERROR: failed to read image from file: " << args.fileInputImage
              << std::endl << err << std::endl; 
    return EXIT_FAILURE;
  }                

  typename InputImageType::Pointer image;

  image = imReader->GetOutput();
  image->DisconnectPipeline();

  DictionaryType imDictionary = image->GetMetaDataDictionary();

  if ( args.flgVerbose )
  {
    DictionaryType::ConstIterator tagItr = imDictionary.Begin();
    DictionaryType::ConstIterator end = imDictionary.End();
   
    while ( tagItr != end )
    {
      MetaDataStringType::ConstPointer entryvalue = 
        dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
      
      if ( entryvalue )
      {
        std::string tagkey = tagItr->first;
        std::string tagID;
        bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, tagID );
        
        std::string tagValue = entryvalue->GetMetaDataObjectValue();
        
        std::cout << tagkey << " " << tagID <<  ": " << tagValue << std::endl;
      }
      
      ++tagItr;
    }
  }
  

  // Read the DICOM data dictionary to be used

  ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

  typename InputImageReaderType::Pointer dcmReader = InputImageReaderType::New();

  dcmReader->SetImageIO( gdcmImageIO );
  dcmReader->SetFileName( args.fileInputDICOM );
    
  try
  {
    std::cout << std::endl << "Reading DICOM data from file: " << args.fileInputDICOM << std::endl;
    dcmReader->Update();
  }

  catch ( itk::ExceptionObject &err )
  {
    std::cerr << "ERROR: failed to read DICOM data from file: " << args.fileInputDICOM
              << std::endl << err << std::endl; 
    return EXIT_FAILURE;
  }

  DictionaryType dcmDictionary = dcmReader->GetOutput()->GetMetaDataDictionary();

  if ( args.flgVerbose )
  {
    DictionaryType::ConstIterator tagItr = dcmDictionary.Begin();
    DictionaryType::ConstIterator end = dcmDictionary.End();
   
    while ( tagItr != end )
    {
      MetaDataStringType::ConstPointer entryvalue = 
        dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
      
      if ( entryvalue )
      {
        std::string tagkey = tagItr->first;
        std::string tagID;
        bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, tagID );
        
        std::string tagValue = entryvalue->GetMetaDataObjectValue();
        
        std::cout << tagkey << " " << tagID <<  ": " << tagValue << std::endl;
      }
      
      ++tagItr;
    }
  }

  // Write the image
  
  typename OutputImageWriterType::Pointer imWriter = OutputImageWriterType::New();

  imWriter->SetFileName( args.fileOutputImage );
  imWriter->SetInput( image );
  
  gdcmImageIO->SetMetaDataDictionary( dcmDictionary );
  gdcmImageIO->KeepOriginalUIDOn( );

  imWriter->SetImageIO( gdcmImageIO );
  
  imWriter->UseInputMetaDataDictionaryOff();
  
  try
  {
    std::cout << std::endl << "Writing image to file: " << args.fileOutputImage << std::endl;
    imWriter->Update(); 
  }
  catch( itk::ExceptionObject &err ) 
  { 
    std::cerr << "ERROR: failed to write image to file: " << args.fileOutputImage
              << std::endl << err << std::endl; 
    return EXIT_FAILURE;
  }                
  return EXIT_SUCCESS;
}

/**
 * \brief Takes the input image and inverts it using InvertIntensityBetweenMaxAndMinImageFilter
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;

  args.flgVerbose      = flgVerbose;
  args.flgDebug        = flgDebug;
 
  args.fileInputImage  = fileInputImage.c_str();
  args.fileInputDICOM  = fileInputDICOM.c_str();

  args.fileOutputImage = fileOutputImage.c_str();

  std::cout << "Input image:       " << args.fileInputImage << std::endl
            << "Input DICOM data:  " << args.fileInputDICOM << std::endl
            << "Output image:      " << args.fileOutputImage << std::endl;

  // Validate command line args
  if (args.fileInputImage.length() == 0 ||
      args.fileInputDICOM.length() == 0 ||
      args.fileOutputImage.length() == 0)
  {
    std::cerr << "ERROR: Input and output files must all be specified" << std::endl;

    return EXIT_FAILURE;
  }

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileInputImage);
  if (dims != 2 && dims != 3)
  {
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  else if (dims == 2)
  {
    std::cout << "Input is 2D" << std::endl;
  }
  else
  {
    std::cout << "Input is 3D" << std::endl;
  }
   
  int result;

  switch (itk::PeekAtComponentType(args.fileInputImage))
  {
  case itk::ImageIOBase::UCHAR:
    std::cout << "Input is UNSIGNED CHAR" << std::endl;
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
    std::cout << "Input is CHAR" << std::endl;
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
    std::cout << "Input is UNSIGNED SHORT" << std::endl;
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
    std::cout << "Input is SHORT" << std::endl;
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
    std::cout << "Input is UNSIGNED INT" << std::endl;
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
    std::cout << "Input is INT" << std::endl;
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
    std::cout << "Input is UNSIGNED LONG" << std::endl;
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
    std::cout << "Input is LONG" << std::endl;
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
    std::cout << "Input is FLOAT" << std::endl;
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
    std::cout << "Input is DOUBLE" << std::endl;
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
    std::cerr << "ERROR: non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }
  return result;
}
