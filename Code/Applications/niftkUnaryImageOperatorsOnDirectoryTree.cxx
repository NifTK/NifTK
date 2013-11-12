/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

/*!
 * \file niftkUnaryImageOperatorsOnDirectoryTree.cxx 
 * \page niftkUnaryImageOperatorsOnDirectoryTree
 * \section niftkUnaryImageOperatorsOnDirectoryTreeSummary niftkUnaryImageOperatorsOnDirectoryTree
 * 
 * Search for images in a directory and apply one of a selection of unary operators on each image, saving the resulting image in the same of a duplicate directory tree.
 *
 */


#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>

#include <itkLogHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>

#include <itkNegateImageFilter.h>
#include <itkSquareImageFilter.h>
#include <itkSqrtImageFilter.h>
#include <itkAbsImageFilter.h>
#include <itkExpImageFilter.h>
#include <itkLogNonZeroIntensitiesImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkGDCMImageIO.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>

#include <vector>

#include <niftkUnaryImageOperatorsOnDirectoryTreeCLP.h>


namespace fs = boost::filesystem;

typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;



struct arguments
{
  std::string inDirectory;
  std::string outDirectory;
  std::string outPixelType;
  std::string outImageFileFormat;
  std::string strAdd2Suffix;  
  std::string imOperation;
  std::string rescaleIntensities;

  bool flgOverwrite;
  bool flgVerbose;
};


// -------------------------------------------------------------------------
// PrintDictionary()
// -------------------------------------------------------------------------

void PrintDictionary( DictionaryType &dictionary )
{
  DictionaryType::ConstIterator tagItr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();
   
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
};


// -------------------------------------------------------------------------
// GetImageFileSuffix
// -------------------------------------------------------------------------

std::string GetImageFileSuffix( std::string fileName )
{
  std::string suffix;
  std::string compSuffix;

  if ( ( fileName.length() >= 3 ) && 
       ( ( fileName.substr( fileName.length() - 3 ) == std::string( ".gz" ) ) || 
         ( fileName.substr( fileName.length() - 3 ) == std::string( ".GZ" ) ) ) )
  {
    compSuffix = fileName.substr( fileName.length() - 3 );
  }

  else if ( ( fileName.length() >= 4 ) && 
            ( ( fileName.substr( fileName.length() - 4 ) == std::string( ".zip" ) ) || 
              ( fileName.substr( fileName.length() - 4 ) == std::string( ".zip" ) ) ) )
  {
    compSuffix = fileName.substr( fileName.length() - 4 );
  }

  if ( ( fileName.length() >= 4 ) && 
       ( ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".dcm" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".DCM" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".nii" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".NII" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".bmp" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".BMP" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".tif" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".TIF" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".jpg" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".JPG" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".png" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".PNG" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".ima" ) ) ||
         ( fileName.substr( fileName.length() - compSuffix.length() - 4, 4 ) == std::string( ".IMA" ) ) ) )
  {
    suffix = fileName.substr( fileName.length() - compSuffix.length() - 4 );
  }

  else if ( ( fileName.length() >= 5 ) && 
            ( ( fileName.substr( fileName.length() - compSuffix.length() - 5, 5 ) == std::string( ".tiff" ) ) || 
              ( fileName.substr( fileName.length() - compSuffix.length() - 5, 5 ) == std::string( ".TIFF" ) ) ||
              ( fileName.substr( fileName.length() - compSuffix.length() - 5, 5 ) == std::string( ".gipl" ) ) ||
              ( fileName.substr( fileName.length() - compSuffix.length() - 5, 5 ) == std::string( ".GIPL" ) ) ) )
  {
    suffix = fileName.substr( fileName.length() - compSuffix.length() - 5 );
  }

  else if ( ( fileName.length() >= 6 ) && 
            ( ( fileName.substr( fileName.length() - compSuffix.length() - 6, 6 ) == std::string( ".dicom" ) ) ||
              ( fileName.substr( fileName.length() - compSuffix.length() - 6, 6 ) == std::string( ".DICOM" ) ) ) )
  {
    suffix =  fileName.substr( fileName.length() - compSuffix.length() - 6 );
  }

  return suffix;
};


// -------------------------------------------------------------------------
// AddToFileSuffix
// -------------------------------------------------------------------------

std::string AddToFileSuffix( std::string fileName, std::string strAdd2Suffix, std::string suffix )
{
  std::string newSuffix;

  std::cout << "Suffix: '" << suffix << "'" << std::endl;

  newSuffix = strAdd2Suffix + suffix;

  if ( ( fileName.length() >= newSuffix.length() ) && 
       ( fileName.substr( fileName.length() - newSuffix.length() ) != newSuffix ) )
  {
    return fileName.substr( 0, fileName.length() - suffix.length() ) + newSuffix;
  }
  else
  {
    return fileName;
  }
};


// -------------------------------------------------------------------------
// DoMain()
// -------------------------------------------------------------------------

template < unsigned int InputDimension, 
           class OutputPixelType >
int DoMain( arguments args, 
            std::string iterFilename,
            std::string suffix )
{

  std::string fileInputFullPath;
  std::string fileInputRelativePath;
  std::string fileOutputRelativePath;
  std::string fileOutputFullPath;
  std::string dirOutputFullPath;
    
  std::vector< std::string >::iterator iterFileNames;       

  typedef double InternalPixelType;

  typedef itk::Image< InternalPixelType, InputDimension > InternalImageType; 
  typedef itk::Image< OutputPixelType, InputDimension > OutputImageType;

  typedef itk::CastImageFilter<InternalImageType, OutputImageType> CastingFilterType;
  typedef itk::MinimumMaximumImageCalculator<InternalImageType> MinimumMaximumImageCalculatorType;
  typedef itk::RescaleIntensityImageFilter< InternalImageType, InternalImageType > RescalerType;
 
  typedef itk::ImageFileReader< InternalImageType > ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;


  typename ReaderType::Pointer reader = ReaderType::New();
  typename InternalImageType::Pointer image;


  // Read the image

  reader->SetFileName( iterFilename );
  reader->UpdateLargestPossibleRegion();

  image = reader->GetOutput();
  image->DisconnectPipeline();

  DictionaryType dictionary = image->GetMetaDataDictionary();
  
   
  // Set the desired output range (i.e. the same as the input)

  typename MinimumMaximumImageCalculatorType::Pointer 
    imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

  imageRangeCalculator->SetImage( image );
  imageRangeCalculator->Compute();

  typename RescalerType::Pointer intensityRescaler = RescalerType::New();

  if ( args.rescaleIntensities == std::string( "to original range" ) )
  {
    intensityRescaler->SetOutputMinimum( 
      static_cast< InternalPixelType >( imageRangeCalculator->GetMinimum() ) );
    intensityRescaler->SetOutputMaximum( 
      static_cast< InternalPixelType >( imageRangeCalculator->GetMaximum() ) );
  }

  else if ( args.rescaleIntensities == std::string( "to maximum image range" ) )
  {
    intensityRescaler->SetOutputMinimum( itk::NumericTraits<OutputPixelType>::NonpositiveMin() );
    intensityRescaler->SetOutputMaximum( itk::NumericTraits<OutputPixelType>::max() );
  }

  else if ( args.rescaleIntensities == std::string( "to maximum positive image range" ) )
  {
    intensityRescaler->SetOutputMinimum( itk::NumericTraits<OutputPixelType>::ZeroValue() );
    intensityRescaler->SetOutputMaximum( itk::NumericTraits<OutputPixelType>::max() );
  }


  if ( args.imOperation 
       == std::string( "invert the image intensities" ) )
  {
    std::cout << "Inverting the image intensities" << std::endl;

    typedef itk::InvertIntensityBetweenMaxAndMinImageFilter<InternalImageType> InvertFilterType;

    typename InvertFilterType::Pointer invfilter = InvertFilterType::New();
    invfilter->SetInput(image);
    invfilter->UpdateLargestPossibleRegion();
    image = invfilter->GetOutput();
  }

  else if ( args.imOperation 
            == std::string( "negate the image intensities" ) )
  {
    std::cout << "Negating the image intensities" << std::endl;

    typedef itk::NegateImageFilter<InternalImageType, InternalImageType> NegateFilterType;

    typename NegateFilterType::Pointer negatefilter = NegateFilterType::New();
    negatefilter->SetInput(image);
    negatefilter->UpdateLargestPossibleRegion();
    image = negatefilter->GetOutput();
  }

  else if ( args.imOperation 
            == std::string( "square the image intensities" ) )
  {
    std::cout << "Computing the square of the intensities" << std::endl;

    typedef itk::SquareImageFilter<InternalImageType, InternalImageType> SquareFilterType;

    typename SquareFilterType::Pointer squarefilter = SquareFilterType::New();
    squarefilter->SetInput(image);
    squarefilter->UpdateLargestPossibleRegion();
    image = squarefilter->GetOutput();
  }

  else if ( args.imOperation 
            == std::string( "square root the image intensities" ) )
  {
    std::cout << "Computing the square root of intensities" << std::endl;

    typedef itk::SqrtImageFilter<InternalImageType, InternalImageType> SqrtFilterType;

    typename SqrtFilterType::Pointer sqrtfilter = SqrtFilterType::New();
    sqrtfilter->SetInput(image);
    sqrtfilter->UpdateLargestPossibleRegion();
    image = sqrtfilter->GetOutput();
  }

  else if ( args.imOperation 
            == std::string( "absolute intensity values" ) )
  {
    std::cout << "Computing the absolute value of intensities" << std::endl;

    typedef itk::AbsImageFilter<InternalImageType, InternalImageType> AbsFilterType;

    typename AbsFilterType::Pointer absfilter = AbsFilterType::New();
    absfilter->SetInput(image);
    absfilter->UpdateLargestPossibleRegion();
    image = absfilter->GetOutput();
  }

  else if ( args.imOperation 
            == std::string( "exponential of intensity values" ) )
  {
    std::cout << "Computing the exponential of the intensities" << std::endl;

    typedef itk::ExpImageFilter<InternalImageType, InternalImageType> ExpFilterType;

    typename ExpFilterType::Pointer expfilter = ExpFilterType::New();
    expfilter->SetInput(image);
    expfilter->UpdateLargestPossibleRegion();
    image = expfilter->GetOutput();
  }

  else if ( args.imOperation 
            == std::string( "natural logarithm of intensity values" ) )
  {
    std::cout << "Computing the natural logarithm of intensities values" << std::endl;

    typedef itk::LogNonZeroIntensitiesImageFilter<InternalImageType, InternalImageType> LogFilterType;

    typename LogFilterType::Pointer logfilter = LogFilterType::New();
    logfilter->SetInput(image);
    logfilter->UpdateLargestPossibleRegion();
    image = logfilter->GetOutput();
  }

  else if ( args.imOperation 
            == std::string( "log-inverse of intensity values" ) )
  {
    std::cout << "Computing the log-inverse of intensities" << std::endl;

    typedef itk::LogNonZeroIntensitiesImageFilter<InternalImageType, InternalImageType> LogFilterType;

    typename LogFilterType::Pointer logfilter = LogFilterType::New();
    logfilter->SetInput(image);
    logfilter->UpdateLargestPossibleRegion();
   
    typedef itk::InvertIntensityBetweenMaxAndMinImageFilter<InternalImageType> InvertFilterType;

    typename InvertFilterType::Pointer invfilter = InvertFilterType::New();
    invfilter->SetInput(logfilter->GetOutput());
    invfilter->UpdateLargestPossibleRegion();
    image = invfilter->GetOutput();
  }

  typename CastingFilterType::Pointer caster = CastingFilterType::New();

  if ( args.rescaleIntensities != std::string( "none" ) )
  {

    std::cout << "Image output range will be: " 
              << intensityRescaler->GetOutputMinimum()
              << " to " << intensityRescaler->GetOutputMaximum() 
              << std::endl;


    intensityRescaler->SetInput( image );  

    intensityRescaler->UpdateLargestPossibleRegion();

    caster->SetInput( intensityRescaler->GetOutput() );
  }
  else
  {
    caster->SetInput( image );
  }

  caster->UpdateLargestPossibleRegion();


  // Create the output image filename

  fileInputFullPath = iterFilename;

  fileInputRelativePath = fileInputFullPath.substr( args.inDirectory.length() );
     
  fileOutputRelativePath = AddToFileSuffix( fileInputRelativePath,
                                            args.strAdd2Suffix, 
                                            suffix );
    
  fileOutputFullPath = niftk::ConcatenatePath( args.outDirectory, 
                                               fileOutputRelativePath );

  dirOutputFullPath = fs::path( fileOutputFullPath ).branch_path().string();
    
  if ( ! niftk::DirectoryExists( dirOutputFullPath ) )
  {
    niftk::CreateDirectoryAndParents( dirOutputFullPath );
  }
      
  std::cout << "Input relative filename: " << fileInputRelativePath << std::endl
            << "Output relative filename: " << fileOutputRelativePath << std::endl
            << "Output directory: " << dirOutputFullPath << std::endl;


  // Write the image to the output file

  if ( niftk::FileExists( fileOutputFullPath ) && ( ! args.flgOverwrite ) )
  {
    std::cerr << std::endl << "ERROR: File " << fileOutputFullPath << " exists"
              << std::endl << "       and can't be overwritten. Consider option: 'overwrite'."
              << std::endl << std::endl;
    return EXIT_FAILURE;
  }
  else
  {
  
    if ( args.flgVerbose )
    {
      PrintDictionary( dictionary );
    }

    typename WriterType::Pointer writer = WriterType::New();

    typename OutputImageType::Pointer outImage = caster->GetOutput();

    writer->SetFileName( fileOutputFullPath );

    outImage->DisconnectPipeline();
    writer->SetInput( outImage );

    typename itk::ImageIOBase::Pointer imageIO;
    imageIO = itk::ImageIOFactory::CreateImageIO(fileOutputFullPath.c_str(), 
						 itk::ImageIOFactory::WriteMode);

    imageIO->SetMetaDataDictionary( dictionary );

    writer->SetImageIO( imageIO );
    writer->UseInputMetaDataDictionaryOff();

    std::cout << "Writing image to file: " 
              << fileOutputFullPath << std::endl;

    writer->Update();
  }

  std::cout << std::endl;


  return EXIT_SUCCESS;
}



// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  float progress = 0.;
  float iFile = 0.;
  float nFiles;

  struct arguments args;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( inDirectory.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    std::cerr << "ERROR: The input directory must be specified" << std::endl;
    return EXIT_FAILURE;
  }

  if ( outDirectory.length() == 0 )
  {
    outDirectory = inDirectory;
  }

  args.inDirectory  = inDirectory;                     
  args.outDirectory = outDirectory;                    

  args.outPixelType = outPixelType;                    
  args.outImageFileFormat = outImageFileFormat;                    

  args.imOperation = imOperation;                    

  args.strAdd2Suffix = strAdd2Suffix;                      
				   	                                                 
  args.flgOverwrite            = flgOverwrite;                       
  args.flgVerbose              = flgVerbose;    

  args.rescaleIntensities = rescaleIntensities;


  std::cout << std::endl << "Examining directory: " 
	    << args.inDirectory << std::endl << std::endl;


  // Get the list of files in the directory
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string iterFilename;
  std::vector< std::string > fileNames;
  std::vector< std::string >::iterator iterFileNames;       

  niftk::GetRecursiveFilesInDirectory( inDirectory, fileNames );

  nFiles = fileNames.size();

  for ( iterFileNames = fileNames.begin(); 
	iterFileNames < fileNames.end(); 
	++iterFileNames, iFile += 1. )
  {
    iterFilename = *iterFileNames;

    std::cout << "File: " << iterFilename << std::endl;

    progress = iFile/nFiles;
    std::cout << "<filter-progress>" << std::endl
	      << progress << std::endl
	      << "</filter-progress>" << std::endl;

    try
    {
  
      itk::ImageIOBase::Pointer imageIO;
      imageIO = itk::ImageIOFactory::CreateImageIO(iterFilename.c_str(), 
                                                   itk::ImageIOFactory::ReadMode);

      if ( ( ! imageIO ) || ( ! imageIO->CanReadFile( iterFilename.c_str() ) ) )
      {
        std::cerr << "WARNING: Unrecognised image type, skipping file: " 
                  << iterFilename << std::endl;
        continue;
      }


      unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels(iterFilename);

      if (dims != 3 && dims != 2)
      {
        std::cout << "WARNING: Unsupported image dimension (" << dims << ") for file: " 
                  << iterFilename << std::endl;
        continue;
      }


      // Determine the desired pixel output type
    
      itk::ImageIOBase::IOComponentType outComponentType;

      if ( args.outPixelType == std::string( "unchanged" ) )
      {
        outComponentType = itk::PeekAtComponentType(iterFilename);
      }
      else if ( args.outPixelType == std::string( "unsigned char" ) )
      {
        outComponentType = itk::ImageIOBase::UCHAR;
      }
      else if ( args.outPixelType == std::string( "char" ) )
      {
        outComponentType = itk::ImageIOBase::CHAR;
      }
      else if ( args.outPixelType == std::string( "unsigned short" ) )
      {
        outComponentType = itk::ImageIOBase::USHORT;            
      }
      else if ( args.outPixelType == std::string( "short" ) )
      {
        outComponentType = itk::ImageIOBase::SHORT;
      }
      else if ( args.outPixelType == std::string( "unsigned int" ) )
      {
        outComponentType = itk::ImageIOBase::UINT;
      }
      else if ( args.outPixelType == std::string( "int" ) )
      {
        outComponentType = itk::ImageIOBase::INT;
      }
      else if ( args.outPixelType == std::string( "unsigned long" ) )
      {
        outComponentType = itk::ImageIOBase::ULONG;
      }
      else if ( args.outPixelType == std::string( "long" ) )
      {
        outComponentType = itk::ImageIOBase::LONG;
      }
      else if ( args.outPixelType == std::string( "float" ) )
      {
        outComponentType = itk::ImageIOBase::FLOAT;
      }
      else if ( args.outPixelType == std::string( "double" ) )
      {
        outComponentType = itk::ImageIOBase::DOUBLE;
      }
      else
      {
        std::cerr << "WARNING: Unrecognised pixel type, skipping file: " 
                  << iterFilename << std::endl;
        continue;
      }
    
      // Get the desired output image file format suffix

      std::string outSuffix;
    
      if ( args.outImageFileFormat == std::string( "unchanged" ) )
      {
        outSuffix = GetImageFileSuffix( iterFilename );
      }
      else if ( args.outImageFileFormat == std::string( "DICOM (.dcm)" ) )
      {
        outSuffix = ".dcm";
      }
      else if ( args.outImageFileFormat == std::string( "Nifti (.nii)" ) )
      {
        outSuffix = ".nii";
      }
      else if ( args.outImageFileFormat == std::string( "GIPL (.gipl)" ) )
      {
        outSuffix = ".gipl";
      }
      else if ( args.outImageFileFormat == std::string( "Bitmap (.bmp)" ) )
      {
        outSuffix = ".bmp";
      }
      else if ( args.outImageFileFormat == std::string( "JPEG (.jpg)" ) )
      {
        outSuffix = ".jpg";
      }
      else if ( args.outImageFileFormat == std::string( "TIFF (.tiff)" ) )
      {
        outSuffix = ".tiff";
      }
      else if ( args.outImageFileFormat == std::string( "PNG (.png)" ) )
      {
        outSuffix = ".png";
      }



      // Operate on this image

      int result;

      switch ( dims )
      {
      case 2:
      {
        switch ( outComponentType )
        {
        case itk::ImageIOBase::UCHAR:
          result = DoMain<2, unsigned char>( args,
                                             iterFilename,
                                             outSuffix );  
          break;
    
        case itk::ImageIOBase::CHAR:
          result = DoMain<2, char>( args,
                                    iterFilename,
                                    outSuffix );  
          break;

        case itk::ImageIOBase::USHORT:
          result = DoMain<2, unsigned short>( args,
                                              iterFilename,
                                              outSuffix );
          break;

        case itk::ImageIOBase::SHORT:
          result = DoMain<2, short>( args,
                                     iterFilename,
                                     outSuffix );
          break;

        case itk::ImageIOBase::UINT:
          result = DoMain<2, unsigned int>( args,
                                            iterFilename,
                                            outSuffix );
          break;

        case itk::ImageIOBase::INT:
          result = DoMain<2, int>( args,
                                   iterFilename,
                                   outSuffix );
          break;

        case itk::ImageIOBase::ULONG:
          result = DoMain<2, unsigned long>( args,
                                             iterFilename,
                                             outSuffix );
          break;

        case itk::ImageIOBase::LONG:
          result = DoMain<2, long>( args,
                                    iterFilename,
                                    outSuffix );
          break;

        case itk::ImageIOBase::FLOAT:
          result = DoMain<2, float>( args,
                                     iterFilename,
                                     outSuffix );
          break;

        case itk::ImageIOBase::DOUBLE:
          result = DoMain<2, double>( args,
                                      iterFilename,
                                      outSuffix );
          break;

        default:
          std::cerr << "WARNING: Unrecognised pixel type, skipping file: " 
                    << iterFilename << std::endl;
        }
        break;
      }

      case 3:
      {
        switch ( outComponentType )
        {
        case itk::ImageIOBase::UCHAR:
          result = DoMain<3, unsigned char>( args,
                                             iterFilename,
                                             outSuffix );  
          break;
    
        case itk::ImageIOBase::CHAR:
          result = DoMain<3, char>( args,
                                    iterFilename,
                                    outSuffix );  
          break;

        case itk::ImageIOBase::USHORT:
          result = DoMain<3, unsigned short>( args,
                                              iterFilename,
                                              outSuffix );
          break;

        case itk::ImageIOBase::SHORT:
          result = DoMain<3, short>( args,
                                     iterFilename,
                                     outSuffix );
          break;

        case itk::ImageIOBase::UINT:
          result = DoMain<3, unsigned int>( args,
                                            iterFilename,
                                            outSuffix );
          break;

        case itk::ImageIOBase::INT:
          result = DoMain<3, int>( args,
                                   iterFilename,
                                   outSuffix );
          break;

        case itk::ImageIOBase::ULONG:
          result = DoMain<3, unsigned long>( args,
                                             iterFilename,
                                             outSuffix );
          break;

        case itk::ImageIOBase::LONG:
          result = DoMain<3, long>( args,
                                    iterFilename,
                                    outSuffix );
          break;

        case itk::ImageIOBase::FLOAT:
          result = DoMain<3, float>( args,
                                     iterFilename,
                                     outSuffix );
          break;

        case itk::ImageIOBase::DOUBLE:
          result = DoMain<3, double>( args,
                                      iterFilename,
                                      outSuffix );
          break;

        default:
          std::cerr << "WARNING: Unrecognised pixel type, skipping file: " 
                    << iterFilename << std::endl;
        }
      }
    
      default:
      {
        std::cout << "WARNING: Unsupported image dimension (" << dims << ") for file: " 
                  << iterFilename << std::endl;
      }
      }

    }
    catch (itk::ExceptionObject & e)
    {
      std::cout << "ERROR: Skipping file: " << iterFilename 
                << std::endl << e << std::endl;
    }

    std::cout << std::endl;
  }

  progress = iFile/nFiles;
  std::cout << "<filter-progress>" << std::endl
            << progress << std::endl
            << "</filter-progress>" << std::endl;
  
  return EXIT_SUCCESS;
}
 
 

