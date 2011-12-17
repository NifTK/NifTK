
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2011-12-16 15:11:48 +#$
 $Rev:: 8053                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkMammoLogImageFilter.h"
#include "itkInvertIntensityImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkGDCMImageIO.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"

#include "gdcmGlobal.h"


#include <list>
#include <fstream>


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "verbose", NULL, "Output verbose debugging info"},

  {OPT_SWITCH, "patient", NULL, "Include the patient name in the filename"},
  {OPT_SWITCH, "bodypart", NULL, "Include the body part imaged in the filename"},
  {OPT_SWITCH, "laterality", NULL, "Include the breast laterality (i.e. L or R) in the filename"},
  {OPT_SWITCH, "view", NULL, "Include the breast view (i.e. CC or MLO) in the filename"},
  {OPT_SWITCH, "date", NULL, "Include the series date in the filename"},
  {OPT_SWITCH, "time", NULL, "Include the series time in the filename"},
  {OPT_SWITCH, "angle", NULL, "Include the positioner primary angle in the filename"},
  {OPT_SWITCH, "thickness", NULL, "Include the breast thickness in the filename"},
  {OPT_SWITCH, "number", NULL, "Include the series number in the filename"},

  {OPT_SWITCH, "preinvert", NULL, "Invert the image prior to the applying the log-invert function"},
  {OPT_SWITCH, "loginvert", NULL, "Log-invert the image (e.g. to convert from Raw to Processed FFDMs)"},

  {OPT_FLOAT, "omin", "minIntensity", "Set the minimum intensity of the output [default is input min]"},
  {OPT_FLOAT, "omax", "maxIntensity", "Set the maximum intensity of the output [default is input max]"},

  {OPT_STRING, "op", "filename", "The output image path and file stem including directory etc."},
  {OPT_STRING, "os", "filename", "The output file suffix e.g. '.nii'."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to log-invert a mammogram and rename it according to whether it is left or right, CC or MLO etc..\n"
  }
};


enum {

  O_VERBOSE,

  O_PATIENT_NAME,
  O_BODY_PART,
  O_LATERALITY,
  O_VIEW,
  O_DATE,
  O_TIME,
  O_ANGLE,
  O_THICKNESS,
  O_NUMBER,    

  O_PRE_INVERT,
  O_LOG_INVERT,

  O_OUTPUT_MIN_INTENSITY,
  O_OUTPUT_MAX_INTENSITY,

  O_OUTPUT_FILE_STEM,
  O_OUTPUT_FILE_SUFFIX,

  O_INPUT_IMAGE
};

typedef itk::MetaDataDictionary   DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;

// -----------------------------------------------------------------------------
// AppendTag()
// -----------------------------------------------------------------------------

void AppendTag( std::string &fileOutputFilename, const DictionaryType &dictionary, 
		std::string entryId, bool flgVerbose )
{
  DictionaryType::ConstIterator tagItr;
  DictionaryType::ConstIterator end = dictionary.End();

  //  It is also possible to read a specific tag. In that case the string of the
  //  entry can be used for querying the MetaDataDictionary.

  tagItr = dictionary.Find( entryId );

  // If the entry is actually found in the Dictionary, then we can attempt to
  // convert it to a string entry by using a \code{dynamic\_cast}.

  if( tagItr != end )
    {
      MetaDataStringType::ConstPointer entryvalue = 
	dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

      // If the dynamic cast succeed, then we can print out the values of the label,
      // the tag and the actual value.
	
      if( entryvalue )
	{
	  std::string tagvalue = entryvalue->GetMetaDataObjectValue();

	  if (flgVerbose) 
	    std::cout << "Tag (" << entryId <<  ") "
		      << " is: " << tagvalue.c_str() << std::endl;
	  
	  fileOutputFilename += tagvalue;
	  fileOutputFilename += "_";
	}
    }
}


// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------

int main( int argc, char * argv[] )
{
  bool flgVerbose;
  bool flgPatientName;
  bool flgBodyPart;
  bool flgLaterality;
  bool flgView;
  bool flgSeriesDate;
  bool flgSeriesTime;
  bool flgPositionerPrimaryAngle;
  bool flgBodyPartThickness;
  bool flgSeriesNumber;

  bool flgPreInvert;
  bool flgLogInvert;

  bool flgMinOutputImageIntensitySet;
  bool flgMaxOutputImageIntensitySet;

  float minInputImageIntensity;
  float maxInputImageIntensity;

  float minOutputImageIntensity;
  float maxOutputImageIntensity;

  std::string fileOutputFilename;

  std::string fileOutputTextFile;
  std::string fileOutputFileStem;
  std::string fileOutputFileSuffix;

  std::string fileInputImage;

  typedef float InternalPixelType;
  typedef signed short OutputPixelType;

  const unsigned int   InputDimension = 2;

  typedef itk::Image< InternalPixelType,  InputDimension > InternalImageType;
  typedef itk::Image< OutputPixelType, InputDimension > OutputImageType;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, flgVerbose );

  CommandLineOptions.GetArgument( O_PATIENT_NAME, flgPatientName );
  CommandLineOptions.GetArgument( O_BODY_PART, flgBodyPart );
  CommandLineOptions.GetArgument( O_LATERALITY, flgLaterality );
  CommandLineOptions.GetArgument( O_VIEW, flgView );
  CommandLineOptions.GetArgument( O_DATE, flgSeriesDate );
  CommandLineOptions.GetArgument( O_TIME, flgSeriesTime );
  CommandLineOptions.GetArgument( O_ANGLE, flgPositionerPrimaryAngle );
  CommandLineOptions.GetArgument( O_THICKNESS, flgBodyPartThickness );
  CommandLineOptions.GetArgument( O_NUMBER, flgSeriesNumber );

  CommandLineOptions.GetArgument( O_PRE_INVERT, flgPreInvert );
  CommandLineOptions.GetArgument( O_LOG_INVERT, flgLogInvert );

  flgMinOutputImageIntensitySet = 
    CommandLineOptions.GetArgument( O_OUTPUT_MIN_INTENSITY, minOutputImageIntensity );

  flgMaxOutputImageIntensitySet =
    CommandLineOptions.GetArgument( O_OUTPUT_MAX_INTENSITY, maxOutputImageIntensity );

  if (CommandLineOptions.GetArgument( O_OUTPUT_FILE_STEM, fileOutputFileStem))
    fileOutputFilename = fileOutputFileStem;

  CommandLineOptions.GetArgument( O_OUTPUT_FILE_SUFFIX, fileOutputFileSuffix );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE, fileInputImage );


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageFileReader< InternalImageType > ReaderType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( fileInputImage );

  
  // GDCMImageIO is an ImageIO class for reading and writing DICOM v3 and
  // ACR/NEMA images. The GDCMImageIO object is constructed here and connected to
  // the ImageFileReader. 

  typedef itk::GDCMImageIO ImageIOType;

  ImageIOType::Pointer gdcmImageIO = ImageIOType::New();
  
  // Here we override the gdcm default value of 0xfff with a value of 0xffff
  // to allow the loading of long binary stream in the DICOM file.
  // This is particularly useful when reading the private tag: 0029,1010
  // from Siemens as it allows to completely specify the imaging parameters
  gdcmImageIO->SetMaxSizeLoadEntry(0xffff);

  reader->SetImageIO( gdcmImageIO );

  try
    {
    reader->Update();
    }
  catch (itk::ExceptionObject & e)
    {
    std::cerr << "exception in file reader " << std::endl;
    std::cerr << e << std::endl;
    return EXIT_FAILURE;
    }

  // Now that the image has been read, we obtain the Meta data dictionary from
  // the ImageIO object using the GetMetaDataDictionary() method.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Now that the image has been read, we obtain the Meta data dictionary from
  // the ImageIO object using the \code{GetMetaDataDictionary()} method.

  const  DictionaryType & dictionary = gdcmImageIO->GetMetaDataDictionary();

  // We instantiate the iterators that will make possible to walk through all the
  // entries of the MetaDataDictionary.

  DictionaryType::ConstIterator itr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();

  // For each one of the entries in the dictionary, we check first if its element
  // can be converted to a string, a \code{dynamic\_cast} is used for this purpose.

  while ( flgVerbose && itr != end ) {

    itk::MetaDataObjectBase::Pointer  entry = itr->second;

    MetaDataStringType::Pointer entryvalue = 
      dynamic_cast<MetaDataStringType *>( entry.GetPointer() );

    // For those entries that can be converted, we take their DICOM tag and pass it
    // to the \code{GetLabelFromTag()} method of the GDCMImageIO class. This method
    // checks the DICOM dictionary and returns the string label associated to the
    // tag that we are providing in the \code{tagkey} variable. If the label is
    // found, it is returned in \code{labelId} variable. The method itself return
    // false if the tagkey is not found in the dictionary.  For example "0010|0010"
    // in \code{tagkey} becomes "Patient's Name" in \code{labelId}.

    if( entryvalue )
      {
      std::string tagkey   = itr->first;
      std::string labelId;
      bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, labelId );

      // The actual value of the dictionary entry is obtained as a string with the
      // \code{GetMetaDataObjectValue()} method.

      std::string tagvalue = entryvalue->GetMetaDataObjectValue();

      // At this point we can print out an entry by concatenating the DICOM Name or
      // label, the numeric tag and its actual value.

      if( found )
        {
        std::cout << "(" << tagkey << ") " << labelId;
        std::cout << " = " << tagvalue.c_str() << std::endl;
        }

      else
        {
        std::cout << "(" << tagkey <<  ") " << "Unknown";
        std::cout << " = " << tagvalue.c_str() << std::endl;
        }
      }

    // Finally we just close the loop that will walk through all the Dictionary
    // entries.

    ++itr;
  }


  if (flgPatientName)
    AppendTag( fileOutputFilename, dictionary, "0010|0010", flgVerbose );

  if (flgSeriesDate)
    AppendTag( fileOutputFilename, dictionary, "0008|0021", flgVerbose );

  if (flgSeriesTime)
    AppendTag( fileOutputFilename, dictionary, "0008|0031", flgVerbose );

  if (flgLaterality) 
    AppendTag( fileOutputFilename, dictionary, "0020|0062", flgVerbose );

  if (flgBodyPart) 
    AppendTag( fileOutputFilename, dictionary, "0018|0015", flgVerbose );

  if (flgView) 
    AppendTag( fileOutputFilename, dictionary, "0018|5101", flgVerbose );

  if (flgPositionerPrimaryAngle)
    AppendTag( fileOutputFilename, dictionary, "0018|1510", flgVerbose );

  if (flgBodyPartThickness)
    AppendTag( fileOutputFilename, dictionary, "0018|11a0", flgVerbose );

  if (flgSeriesNumber)
    AppendTag( fileOutputFilename, dictionary, "0020|0011", flgVerbose );

  // Remove white space
  
  std::string::size_type idx;
  
  idx = fileOutputFilename.find(" ");
  while (idx != std::string::npos) {
    fileOutputFilename.erase(idx, 1);
    idx = fileOutputFilename.find(" ");
  }


  // Create the filename
  // ~~~~~~~~~~~~~~~~~~~

  fileOutputTextFile = fileOutputFilename + ".txt";

  if (fileOutputFileSuffix.length() > 0)
    fileOutputFilename += fileOutputFileSuffix;


  // Write the image
  // ~~~~~~~~~~~~~~~

  if ((fileOutputFileStem.length() != 0) || (fileOutputFileSuffix.length() != 0)) {

    typedef itk::MinimumMaximumImageCalculator< InternalImageType > MinimumMaximumImageCalculatorType;
    MinimumMaximumImageCalculatorType::Pointer imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

    typedef itk::InvertIntensityImageFilter< InternalImageType, InternalImageType > InvertIntensityImageFilterType;

    typedef itk::ImageFileWriter< OutputImageType >  WriterType;
    WriterType::Pointer writer = WriterType::New();


    writer->SetFileName( fileOutputFilename );

    InternalImageType::Pointer intermediateImage = reader->GetOutput();;

    imageRangeCalculator->SetImage( intermediateImage );
    imageRangeCalculator->Compute();

    maxInputImageIntensity = imageRangeCalculator->GetMaximum();
    minInputImageIntensity = imageRangeCalculator->GetMinimum();


    if (flgPreInvert) {

      InvertIntensityImageFilterType::Pointer preInvertFilter = InvertIntensityImageFilterType::New();
      
      std::cout << "Pre-inverting image with max intensity: " << imageRangeCalculator->GetMaximum() << std::endl;

      preInvertFilter->SetInput( intermediateImage );
      preInvertFilter->SetMaximum( maxInputImageIntensity );
      preInvertFilter->UpdateLargestPossibleRegion();

      intermediateImage = preInvertFilter->GetOutput();
    }

    if (flgLogInvert) {

      typedef itk::MammoLogImageFilter< InternalImageType, InternalImageType > LogImageFilterType;
      LogImageFilterType::Pointer logFilter = LogImageFilterType::New();
      
      logFilter->SetInput( intermediateImage );
      logFilter->UpdateLargestPossibleRegion();

      InvertIntensityImageFilterType::Pointer postInvertFilter = InvertIntensityImageFilterType::New();
      
      imageRangeCalculator->SetImage( logFilter->GetOutput() );
      imageRangeCalculator->Compute();

      std::cout << "Post-inverting image with max intensity: " << imageRangeCalculator->GetMaximum() << std::endl;

      postInvertFilter->SetInput( logFilter->GetOutput() );
      postInvertFilter->SetMaximum( imageRangeCalculator->GetMaximum() );
      postInvertFilter->UpdateLargestPossibleRegion();

    
      // Rescale outputs to the dynamic range of the display

      typedef itk::RescaleIntensityImageFilter< InternalImageType, InternalImageType > RescalerType;

      RescalerType::Pointer intensityRescaler = RescalerType::New();

      // If the user supplied an output rang use it otherwise use the input values
      if ( flgMinOutputImageIntensitySet )
	intensityRescaler->SetOutputMinimum( static_cast< OutputPixelType >( minOutputImageIntensity ) );
      else
	intensityRescaler->SetOutputMinimum( static_cast< OutputPixelType >( minInputImageIntensity ) );

      if ( flgMaxOutputImageIntensitySet )
	intensityRescaler->SetOutputMaximum( static_cast< OutputPixelType >( maxOutputImageIntensity ) );
      else
	intensityRescaler->SetOutputMaximum( static_cast< OutputPixelType >( maxInputImageIntensity ) );

      intensityRescaler->SetInput( postInvertFilter->GetOutput() );  
      intensityRescaler->UpdateLargestPossibleRegion();

      intermediateImage = intensityRescaler->GetOutput();
    }

    // Cast the image to the output type

    typedef itk::CastImageFilter< InternalImageType, OutputImageType > CastingFilterType;

    CastingFilterType::Pointer caster = CastingFilterType::New();

    caster->SetInput( intermediateImage );
    caster->UpdateLargestPossibleRegion();

    writer->SetInput( caster->GetOutput() );

    try
      {
	std::cout << "Writing the output image to: " << fileOutputFilename << std::endl << std::endl;
	writer->Update();
      }
    catch (itk::ExceptionObject &e)
      {
	std::cerr << e << std::endl;
      }


    // Write the text file
    // ~~~~~~~~~~~~~~~~~~~

    std::fstream fout;

    fout.open(fileOutputTextFile.c_str(), std::ios::out);
    
    if ((! fout) || fout.bad()) {
      std::cerr << "Failed to open file: " << fileOutputTextFile.c_str() << std::endl;
      return EXIT_FAILURE;      
    }

    fout << fileInputImage << std::endl << std::endl;
    
    itr = dictionary.Begin();
    end = dictionary.End();

    // For each one of the entries in the dictionary, we check first if its element
    // can be converted to a string, a \code{dynamic\_cast} is used for this purpose.

    while ( itr != end ) {

      itk::MetaDataObjectBase::Pointer  entry = itr->second;

      MetaDataStringType::Pointer entryvalue = 
	dynamic_cast<MetaDataStringType *>( entry.GetPointer() );

      if( entryvalue ) {
	std::string tagkey   = itr->first;
	std::string labelId;
	bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, labelId );

	std::string tagvalue = entryvalue->GetMetaDataObjectValue();

	if( found ) {
	  fout << "(" << tagkey << ") " << labelId;
	  fout << " = " << tagvalue.c_str() << std::endl;
        }
	
	else {
	  fout << "(" << tagkey <<  ") " << "Unknown";
	  fout << " = " << tagvalue.c_str() << std::endl;
        }
      }

      ++itr;
    }

    fout.close();


  }
  else
    std::cout << fileOutputFilename << std::endl;



  return EXIT_SUCCESS;

}
