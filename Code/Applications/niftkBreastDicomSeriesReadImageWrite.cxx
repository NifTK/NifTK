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

#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkImageFileWriter.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkOrientImageFilter.h>

#include <gdcmGlobal.h>

#include <vector>



struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_STRING, "series", "name", "The input series name required."},
  {OPT_STRING, "filter", "pattern", "Only consider DICOM files that contain the string 'pattern'."},

  {OPT_INT, "orient", "value", 
   "Orient the image according to itk::SpatialOrientation::ValidCoordinateOrientationFlags.\n"
   "\t e.g. RAI (19): x axis runs from R to L, y from A to P and z from I to S.\n"
   "\t  1=RIP,  2=LIP,  3=RSP,  4=LSP,  5=RIA,  6=LIA,  7=RSA,  8=LSA,\n"
   "\t  9=IRP, 10=ILP, 11=SRP, 12=SLP, 13=IRA, 14=ILA, 15=SRA, 16=SLA,\n"
   "\t 17=RPI, 18=LPI, 19=RAI, 20=LAI, 21=RPS, 22=LPS, 23=RAS, 24=LAS,\n"
   "\t 25=PRI, 26=PLI, 27=ARI, 28=ALI, 29=PRS, 30=PLS, 31=ARS, 32=ALS,\n"
   "\t 33=IPR, 34=SPR, 35=IAR, 36=SAR, 37=IPL, 38=SPL, 39=IAL, 40=SAL,\n"
   "\t 41=PIR, 42=PSR, 43=AIR, 44=ASR, 45=PIL, 46=PSL, 47=AIL, 48=ASL."
  },

  {OPT_STRING, "of", "filestem", "The output filestem or directory (filename will be: 'filestem%s.suffix')."},
  {OPT_STRING|OPT_REQ, "os", "suffix",   "The output image suffix to use when using option '-of'."},

  {OPT_STRING|OPT_REQ|OPT_LONELY, NULL, "directory", "Input DICOM directory."},

  {OPT_DONE, NULL, NULL, 
   "Program to convert the content of a DICOM directory into breast image volumes.\n"
  }
};

enum { 
  O_VERBOSE,
  O_DEBUG,

  O_SERIES,
  O_FILENAME_FILTER,

  O_ORIENTATION,

  O_OUTPUT_FILESTEM,
  O_OUTPUT_FILESUFFIX,

  O_INPUT_DICOM_DIRECTORY
};


struct arguments
{
  // Set up defaults
  arguments()
  {
    flgVerbose = false;
    flgDebug = false;
    orientation = 0;
  }

  bool flgVerbose;
  bool flgDebug;
  
  int orientation;

  std::string seriesName;
  std::string fileNameFilter;

  std::string fileOutputStem;
  std::string fileOutputSuffix;

  std::string dirDICOMInput;
};
  
const char *orientationCodeName[49] = {
  "INVALID",
  "RIP", //  1
  "LIP", //  2
  "RSP", //  3
  "LSP", //  4
  "RIA", //  5
  "LIA", //  6
  "RSA", //  7
  "LSA", //  8
  "IRP", //  9
  "ILP", // 10
  "SRP", // 11
  "SLP", // 12
  "IRA", // 13
  "ILA", // 14
  "SRA", // 15
  "SLA", // 16
  "RPI", // 17
  "LPI", // 18
  "RAI", // 19
  "LAI", // 20
  "RPS", // 21
  "LPS", // 22
  "RAS", // 23
  "LAS", // 24
  "PRI", // 25
  "PLI", // 26
  "ARI", // 27
  "ALI", // 28
  "PRS", // 29
  "PLS", // 30
  "ARS", // 31
  "ALS", // 32
  "IPR", // 33
  "SPR", // 34
  "IAR", // 35
  "SAR", // 36
  "IPL", // 37
  "SPL", // 38
  "IAL", // 39
  "SAL", // 40
  "PIR", // 41
  "PSR", // 42
  "AIR", // 43
  "ASR", // 44
  "PIL", // 45
  "PSL", // 46
  "AIL", // 47
  "ASL", // 48
};

// We define the pixel type and dimension of the image to be read. In this
// particular case, the dimensionality of the image is 3, and we assume a
// \code{signed short} pixel type that is commonly used for X-Rays CT scanners.

typedef signed short    PixelType;
const unsigned int      Dimension = 3;

typedef itk::Image< PixelType, Dimension > ImageType;

// We use the image type for instantiating the type of the series reader and
// for constructing one object of its type.

typedef itk::ImageSeriesReader< ImageType > ReaderType;

typedef itk::GDCMSeriesFileNames NamesGeneratorType;

typedef itk::MetaDataDictionary   DictionaryType;

DictionaryType::ConstIterator tagItr;

// Since we are interested only in the DICOM tags that can be expressed in
// strings, we declare a MetaDataObject suitable for managing strings.

typedef itk::MetaDataObject< std::string > MetaDataStringType;

typedef std::vector< std::string > FileNamesContainer;

bool WriteSeriesAsVolume( std::fstream &fout, 
			  bool flgVerbose,
			  bool flgDebug,
			  int orientation,
			  std::string seriesIdentifier,
			  std::string filenameFilter,
			  std::string fileOutputStem,
			  std::string fileOutputSuffix,
			  std::string fileOutput,
			  FileNamesContainer &fileNames,
			  ReaderType *reader, 
			  std::string tagModalityValue );


// -----------------------------------------------------------------------------
// DumpDirections()
// -----------------------------------------------------------------------------

typedef itk::SpatialOrientation::ValidCoordinateOrientationFlags 
SO_OrientationType;
std::string SO_OrientationToString(SO_OrientationType in)
{
  switch(in)
    {
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP:
      return std::string("RIP");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIP:
      return std::string("LIP");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSP:
      return std::string("RSP");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSP:
      return std::string("LSP");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIA:
      return std::string("RIA");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIA:
      return std::string("LIA");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSA:
      return std::string("RSA");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSA:
      return std::string("LSA");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRP:
      return std::string("IRP");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILP:
      return std::string("ILP");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRP:
      return std::string("SRP");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLP:
      return std::string("SLP");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRA:
      return std::string("IRA");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILA:
      return std::string("ILA");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRA:
      return std::string("SRA");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLA:
      return std::string("SLA");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI:
      return std::string("RPI");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPI:
      return std::string("LPI");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI:
      return std::string("RAI");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI:
      return std::string("LAI");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPS:
      return std::string("RPS");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPS:
      return std::string("LPS");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS:
      return std::string("RAS");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAS:
      return std::string("LAS");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRI:
      return std::string("PRI");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLI:
      return std::string("PLI");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARI:
      return std::string("ARI");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALI:
      return std::string("ALI");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRS:
      return std::string("PRS");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLS:
      return std::string("PLS");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARS:
      return std::string("ARS");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALS:
      return std::string("ALS");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPR:
      return std::string("IPR");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPR:
      return std::string("SPR");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAR:
      return std::string("IAR");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAR:
      return std::string("SAR");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPL:
      return std::string("IPL");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPL:
      return std::string("SPL");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAL:
      return std::string("IAL");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAL:
      return std::string("SAL");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR:
      return std::string("PIR");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSR:
      return std::string("PSR");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIR:
      return std::string("AIR");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASR:
      return std::string("ASR");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIL:
      return std::string("PIL");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSL:
      return std::string("PSL");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL:
      return std::string("AIL");
    case itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASL:
      return "ASL";
    default:
      {
      std::stringstream x;
      x << (in & 0xff) << ", " << ((in >> 8) & 0xff) << ", " << ((in >> 16) && 0xff);
      return x.str();
      }
    }
}


// -----------------------------------------------------------------------------
// DumpDirections()
// -----------------------------------------------------------------------------

template <class ImageType>
void
DumpDirections(const std::string &prompt, const typename ImageType::Pointer &image)
{
  const typename ImageType::DirectionType &dir =
    image->GetDirection();
  std::cerr << prompt << " " 
            << SO_OrientationToString(itk::SpatialOrientationAdapter().FromDirectionCosines(dir))
            <<    std::endl;
  for(unsigned i = 0; i < 3; i++)
    {
    for(unsigned j = 0; j < 3; j++)
      {
      std::cerr << dir[i][j] << " ";
      }
    std::cerr << std::endl;
    }
}


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
// SeriesOutputFilename()
// -----------------------------------------------------------------------------

std::string SeriesOutputFilename( std::fstream &fout, std::string fileInputImage, 
				  std::string fileOutputStem, bool flgVerbose, 
				  std::string &tagModalityValue )
{
  std::string entryId;
  std::string fileOutputFilename;

  std::string tagModalityKey("0008|0060");

  tagModalityValue = " ";


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageFileReader< ImageType > ReaderType;

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

  std::cout << "SeriesOutputFilename: Reading image..." << std::endl;

  try
    {
    reader->UpdateLargestPossibleRegion();
    }
  catch (itk::ExceptionObject & e)
    {
    std::cerr << "exception in file reader " << std::endl;
    std::cerr << e << std::endl;
    exit( EXIT_FAILURE );
    }

  std::cout << "SeriesOutputFilename: done." << std::endl;


  // Now that the image has been read, we obtain the Meta data dictionary from
  // the ImageIO object using the GetMetaDataDictionary() method.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  const  DictionaryType & dictionary = gdcmImageIO->GetMetaDataDictionary();


  // Patient name
  AppendTag( fileOutputFilename, dictionary, "0010|0010", flgVerbose );

  // Modality
  AppendTag( fileOutputFilename, dictionary, "0008|0060", flgVerbose );

  // Series date
  AppendTag( fileOutputFilename, dictionary, "0008|0021", flgVerbose );

  // Series time
  AppendTag( fileOutputFilename, dictionary, "0008|0031", flgVerbose );

  // Series description
  AppendTag( fileOutputFilename, dictionary, "0008|103e", flgVerbose );

  // Protocol name
  //AppendTag( fileOutputFilename, dictionary, "0018|1030", flgVerbose );

  // Body part
  AppendTag( fileOutputFilename, dictionary, "0018|0015", flgVerbose );

  // Breast view
  AppendTag( fileOutputFilename, dictionary, "0018|5101", flgVerbose );

  // Breast laterality
  AppendTag( fileOutputFilename, dictionary, "0020|0062", flgVerbose );

  // Positioner Primary Angle
  AppendTag( fileOutputFilename, dictionary, "0018|1510", flgVerbose );

  // Body part thickness
  AppendTag( fileOutputFilename, dictionary, "0018|11a0", flgVerbose );

  // Series number
  AppendTag( fileOutputFilename, dictionary, "0020|0011", flgVerbose );

  // Remove white space
  
  std::string::size_type idx;

  idx = fileOutputFilename.find(" ");
  while (idx != std::string::npos) {
    fileOutputFilename.erase(idx, 1);
    idx = fileOutputFilename.find(" ");
  }

  // and '/' characters
  
  idx = fileOutputFilename.find("/");
  while (idx != std::string::npos) {
    fileOutputFilename.replace(idx, 1, "-");
    idx = fileOutputFilename.find("/");
  }


  // Write the dicom data for this series to a text file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string fileDicomHeaderOut =fileOutputStem + fileOutputFilename + ".txt";

  fout.open(fileDicomHeaderOut.c_str(), std::ios::out);
    
  if ((! fout) || fout.bad()) {
    std::cerr << "Failed to open file: "
				   << fileDicomHeaderOut.c_str();
    exit( EXIT_FAILURE );
  }

  // We instantiate the iterators that will make possible to walk through all the
  // entries of the MetaDataDictionary.

  DictionaryType::ConstIterator itr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();

  // For each one of the entries in the dictionary, we check first if its element
  // can be converted to a string, a \code{dynamic\_cast} is used for this purpose.

  while ( itr != end ) {

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

    if ( entryvalue ) {

      std::string tagkey   = itr->first;
      std::string labelId;
      bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, labelId );

      // The actual value of the dictionary entry is obtained as a string with the
      // \code{GetMetaDataObjectValue()} method.

      std::string tagvalue = entryvalue->GetMetaDataObjectValue();

      // At this point we can print out an entry by concatenating the DICOM Name or
      // label, the numeric tag and its actual value.

      if ( found ) {

        fout << "(" << tagkey << ") " << labelId;
        fout << " = " << tagvalue.c_str() << std::endl;
      }

      else {

        fout << "(" << tagkey <<  ") " << "Unknown";
        fout << " = " << tagvalue.c_str() << std::endl;
      }

      if ( tagkey == tagModalityKey ) 
	tagModalityValue = tagvalue;
      
    }

    // Finally we just close the loop that will walk through all the Dictionary
    // entries.

    ++itr;
  }
      
  return fileOutputFilename;
}



// -----------------------------------------------------------------------------
// WriteSeriesAsVolume()
// -----------------------------------------------------------------------------

bool WriteSeriesAsVolume( std::fstream &fout, 
			  bool flgVerbose,
			  bool flgDebug,
			  int orientation,
			  std::string seriesIdentifier,
			  std::string filenameFilter,
			  std::string fileOutputStem,
			  std::string fileOutputSuffix,
			  std::string fileOutput,
			  NamesGeneratorType *nameGenerator,
			  ReaderType *reader, 
			  std::string tagModalityValue ) 
{
  std::vector< std::string >::iterator iterFilenames;

  // We pass the series identifier to the name generator and ask for all the
  // filenames associated to that series. This list is returned in a container of
  // strings by the \code{GetFileNames()} method. 
  
  FileNamesContainer fileNames;
  
  fileNames = nameGenerator->GetFileNames( seriesIdentifier );

  // If the user has specified a filename substring then
  // we can use this to filter out the images we don't want

  if ( filenameFilter.length() > 0 ) {

    FileNamesContainer filteredFileNames;
    
    for (iterFilenames=fileNames.begin(); iterFilenames<fileNames.end(); ++iterFilenames) 

      if ( iterFilenames->find( filenameFilter ) != std::string::npos )
	filteredFileNames.push_back( *iterFilenames );

      else if ( flgDebug ) 
	std::cout << "   ignoring file: " << *iterFilenames << std::endl;


    if ( filteredFileNames.size() < 1 )
      return false;
    else
      fileNames = filteredFileNames;
  }


  return WriteSeriesAsVolume( fout, 
			      flgVerbose,
			      flgDebug,
			      orientation,
			      seriesIdentifier,
			      filenameFilter,
			      fileOutputStem,
			      fileOutputSuffix,
			      fileOutput,
			      fileNames,
			      reader, 
			      tagModalityValue );
}


// -----------------------------------------------------------------------------
// WriteSeriesAsVolume()
// -----------------------------------------------------------------------------

bool WriteSeriesAsVolume( std::fstream &fout, 
			  bool flgVerbose,
			  bool flgDebug,
			  int orientation,
			  std::string seriesIdentifier,
			  std::string filenameFilter,
			  std::string fileOutputStem,
			  std::string fileOutputSuffix,
			  std::string fileOutput,
			  FileNamesContainer &fileNames,
			  ReaderType *reader, 
			  std::string tagModalityValue ) 
{
  std::vector< std::string >::iterator iterFilenames;

  if ( flgVerbose )
    std::cout << std::endl << std::endl
	      << "Now reading series: " << std::endl << std::endl
	      << seriesIdentifier << std::endl
	      << std::endl << std::endl;

  fout << std::endl << "Series: " << seriesIdentifier << std::endl << std::endl;
  
  for (iterFilenames=fileNames.begin(); iterFilenames<fileNames.end(); ++iterFilenames) {

    if ( flgVerbose )
      std::cout << "   " << *iterFilenames << std::endl;

    fout << "   " << *iterFilenames << std::endl;  
  }


  // The list of filenames can now be passed to the \doxygen{ImageSeriesReader}
  // using the \code{SetFileNames()} method.
  //  
  //  \index{itk::ImageSeriesReader!SetFileNames()}
  
  reader->SetFileNames( fileNames );

  // Finally we can trigger the reading process by invoking the \code{Update()}
  // method in the reader. This call as usual is placed inside a \code{try/catch}
  // block.

  try
    {
      reader->UpdateLargestPossibleRegion();
    }

  catch (itk::ExceptionObject &ex)
    {
      // If the read failed, output the images individually and exit function
      
      std::cout << ex << std::endl;
      
      int iImage = 0;
      for (iterFilenames=fileNames.begin(); iterFilenames<fileNames.end(); ++iterFilenames) {

	FileNamesContainer individualFiles;
	individualFiles.push_back( *iterFilenames );

	WriteSeriesAsVolume( fout, 
			     flgVerbose,
			     flgDebug,
			     orientation,
			     seriesIdentifier,
			     filenameFilter,
			     fileOutputStem,
			     fileOutputSuffix,
			     fileOutput + "_" + niftk::ConvertToString( iImage ),
			     individualFiles,
			     reader, 
			     tagModalityValue );
	
	iImage++;
      }
      return true;
    }

  ImageType::Pointer intermediateImage = reader->GetOutput();


  // Reorientate the image to a standard orientation?

   if ( (tagModalityValue == "MR") && orientation ) {

     itk::SpatialOrientation::ValidCoordinateOrientationFlags orientationCode;

     switch ( orientation ) 
       {

       case  1: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP; break; }
       case  2: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIP; break; }
       case  3: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSP; break; }
       case  4: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSP; break; }
       case  5: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIA; break; }
       case  6: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIA; break; }
       case  7: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSA; break; }
       case  8: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSA; break; }
       case  9: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRP; break; }
       case 10: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILP; break; }
       case 11: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRP; break; }
       case 12: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLP; break; }
       case 13: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRA; break; }
       case 14: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILA; break; }
       case 15: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRA; break; }
       case 16: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLA; break; }
       case 17: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI; break; }
       case 18: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPI; break; }
       case 19: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI; break; }
       case 20: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI; break; }
       case 21: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPS; break; }
       case 22: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPS; break; }
       case 23: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS; break; }
       case 24: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAS; break; }
       case 25: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRI; break; }
       case 26: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLI; break; }
       case 27: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARI; break; }
       case 28: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALI; break; }
       case 29: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRS; break; }
       case 30: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLS; break; }
       case 31: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARS; break; }
       case 32: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALS; break; }
       case 33: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPR; break; }
       case 34: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPR; break; }
       case 35: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAR; break; }
       case 36: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAR; break; }
       case 37: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPL; break; }
       case 38: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPL; break; }
       case 39: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAL; break; }
       case 40: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAL; break; }
       case 41: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR; break; }
       case 42: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSR; break; }
       case 43: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIR; break; }
       case 44: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASR; break; }
       case 45: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIL; break; }
       case 46: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSL; break; }
       case 47: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL; break; }
       case 48: { orientationCode = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASL; break; }
       default: {
	 std::cerr << "ERROR Unrecognised spatial orientation type: " << orientation << std::endl;
	 exit( EXIT_FAILURE );
       }
       }
     
     itk::OrientImageFilter<ImageType,ImageType>::Pointer orienter =
       itk::OrientImageFilter<ImageType,ImageType>::New();
     
     orienter->UseImageDirectionOn();
     orienter->SetDesiredCoordinateOrientation( orientationCode );
     
     orienter->SetInput( intermediateImage );
     orienter->Update();

     if ( flgVerbose ) 
       std::cout << std::endl
		 << "Input Coordinate Orientation: "
		 << SO_OrientationToString( orienter->GetGivenCoordinateOrientation() )
		 << std::endl
		 << "Output Coordinate Orientation: "
		 << SO_OrientationToString( orienter->GetDesiredCoordinateOrientation() )
		 << std::endl
		 << "Permute Axes: " << orienter->GetPermuteOrder() << std::endl
		 << "Flip Axes: "    << orienter->GetFlipAxes() << std::endl;

     fout << std::endl
	  << "Input Coordinate Orientation: "
	  << SO_OrientationToString( orienter->GetGivenCoordinateOrientation() )
	  << std::endl
	  << "Output Coordinate Orientation: "
	  << SO_OrientationToString( orienter->GetDesiredCoordinateOrientation() )
	  << std::endl
	  << "Permute Axes: " << orienter->GetPermuteOrder() << std::endl
	  << "Flip Axes: "    << orienter->GetFlipAxes() << std::endl;

     intermediateImage = orienter->GetOutput();
   }


  // At this point, we have a volumetric image in memory that we can access by
  // invoking the \code{GetOutput()} method of the reader.
  //
  // We proceed now to save the volumetric image in another file, as specified by
  // the user in the command line arguments of this program. Thanks to the
  // ImageIO factory mechanism, only the filename extension is needed to identify
  // the file format in this case.


  typedef itk::ImageFileWriter< ImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
    
  writer->SetFileName( fileOutputStem + fileOutput + "." + fileOutputSuffix);

  writer->SetInput( intermediateImage );

  std::cout << std::endl << "Writing the image as: " 
	    << fileOutputStem + fileOutput + "." + fileOutputSuffix << std::endl << std::endl;

  // The process of writing the image is initiated by invoking the
  // \code{Update()} method of the writer.

  try
    {
      writer->Update();
    }
  catch (itk::ExceptionObject &ex)
    {
      std::cout << ex << std::endl;
      exit( EXIT_FAILURE );
    }

  return true;
}



// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  std::string fileOutput;
  std::string tagModalityValue;

  // To pass around command line args
  struct arguments args;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

  CommandLineOptions.GetArgument( O_SERIES, args.seriesName );
  CommandLineOptions.GetArgument( O_FILENAME_FILTER, args.fileNameFilter );

  CommandLineOptions.GetArgument( O_ORIENTATION, args.orientation );

  CommandLineOptions.GetArgument( O_OUTPUT_FILESTEM, args.fileOutputStem );
  CommandLineOptions.GetArgument( O_OUTPUT_FILESUFFIX, args.fileOutputSuffix );

  CommandLineOptions.GetArgument( O_INPUT_DICOM_DIRECTORY, args.dirDICOMInput );
  

  ReaderType::Pointer reader = ReaderType::New();
 
  // A GDCMImageIO object is created and connected to the reader. This object is
  // the one that is aware of the internal intricacies of the DICOM format. 

  typedef itk::GDCMImageIO       ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();
  
  reader->SetImageIO( dicomIO );

  // Now we face one of the main challenges of the process of reading a DICOM
  // series. That is, to identify from a given directory the set of filenames
  // that belong together to the same volumetric image. Fortunately for us, GDCM
  // offers functionalities for solving this problem and we just need to invoke
  // those functionalities through an ITK class that encapsulates a communication
  // with GDCM classes. This ITK object is the GDCMSeriesFileNames. Conveniently
  // for us, we only need to pass to this class the name of the directory where
  // the DICOM slices are stored. This is done with the \code{SetDirectory()}
  // method. The GDCMSeriesFileNames object will explore the directory and will
  // generate a sequence of filenames for DICOM files for one study/series. 
  // In this example, we also call the \code{SetUseSeriesDetails(true)} function
  // that tells the GDCMSeriesFileNames object to use additional DICOM 
  // information to distinguish unique volumes within the directory.  This is
  // useful, for example, if a DICOM device assigns the same SeriesID to 
  // a scout scan and its 3D volume; by using additional DICOM information
  // the scout scan will not be included as part of the 3D volume.  Note that
  // \code{SetUseSeriesDetails(true)} must be called prior to calling
  // \code{SetDirectory()}. By default \code{SetUseSeriesDetails(true)} will use
  // the following DICOM tags to sub-refine a set of files into multiple series:
  // * 0020 0011 Series Number
  // * 0018 0024 Sequence Name
  // * 0018 0050 Slice Thickness
  // * 0028 0010 Rows
  // * 0028 0011 Columns
  // If this is not enough for your specific case you can always add some more
  // restrictions using the \code{AddSeriesRestriction()} method. In this example we will use
  // the DICOM Tag: 0008 0021 DA 1 Series Date, to sub-refine each series. The format
  // for passing the argument is a string containing first the group then the element
  // of the DICOM tag, separed by a pipe (|) sign.

  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

  nameGenerator->SetLoadSequences( true );
  nameGenerator->SetLoadPrivateTags( true ); 

  nameGenerator->SetUseSeriesDetails( true );

  nameGenerator->AddSeriesRestriction( "0008|0008" ); // Image type
  nameGenerator->AddSeriesRestriction( "0008|0021" );
  nameGenerator->AddSeriesRestriction( "0008|0032" ) ;// Acquisition Time
  nameGenerator->AddSeriesRestriction( "0008|0033" ); // Content (formerly Image) Time
  nameGenerator->AddSeriesRestriction( "0018|0060" ); // KVp
  nameGenerator->AddSeriesRestriction( "0018|1114" ); // Estimated Radi...cation Factor
  nameGenerator->AddSeriesRestriction( "0018|1150" ); // Exposure Time
  nameGenerator->AddSeriesRestriction( "0018|1151" ); // X-ray Tube Current
  nameGenerator->AddSeriesRestriction( "0018|1152" ); // Exposure
  nameGenerator->AddSeriesRestriction( "0018|1153" ); // Exposure in uAs
  nameGenerator->AddSeriesRestriction( "0018|11A2" ); // Compression Force
  nameGenerator->AddSeriesRestriction( "0018|1510" ); // Positioner Primary Angle
  nameGenerator->AddSeriesRestriction( "0018|5101" ); // Breast view
  nameGenerator->AddSeriesRestriction( "0020|0020" ); // Patient Orientation
  nameGenerator->AddSeriesRestriction( "0028|0010" ); // Image dimensions: number of rows and columns
  nameGenerator->AddSeriesRestriction( "0028|0011" );

  nameGenerator->SetDirectory( args.dirDICOMInput );
  
  try
    {      
    
      // The GDCMSeriesFileNames object first identifies the list of DICOM series
      // that are present in the given directory. We receive that list in a
      // reference to a container of strings and then we can do things like
      // printing out all the series identifiers that the generator had
      // found. Since the process of finding the series identifiers can
      // potentially throw exceptions, it is wise to put this code inside a
      // try/catch block.

      typedef std::vector<std::string> seriesIdContainer;
      const seriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

      seriesIdContainer::const_iterator seriesItr = seriesUID.begin();
      seriesIdContainer::const_iterator seriesEnd = seriesUID.end();
      
      if ( args.flgVerbose ) {
	std::cout << std::endl << "The directory: " << std::endl;
	std::cout << std::endl << args.dirDICOMInput << std::endl << std::endl;
	std::cout << "Contains the following DICOM Series: ";
	std::cout << std::endl << std::endl;
      }

      while( seriesItr != seriesEnd )
	{
	  std::cout << seriesItr->c_str() << std::endl;
	  seriesItr++;
	}

      // Given that it is common to find multiple DICOM series in the same directory,
      // we must tell the GDCM classes what specific series do we want to read. In
      // this example we do this by checking first if the user has provided a series
      // identifier in the command line arguments. 
      
      if ( args.seriesName.length() > 0 ) {

	std::fstream fout;

	fileOutput = SeriesOutputFilename( fout,
					   nameGenerator->GetFileNames( args.seriesName )[0],
					   args.fileOutputStem, args.flgVerbose,
					   tagModalityValue );
      
	std::cout << "Filename: " << fileOutput << std::endl;

	WriteSeriesAsVolume( fout,
			     args.flgVerbose, 
			     args.flgDebug, 
			     args.orientation,
			     args.seriesName, 
			     args.fileNameFilter,
			     args.fileOutputStem, 
			     args.fileOutputSuffix, 
			     fileOutput, 
			     nameGenerator, 
			     reader,
			     tagModalityValue );

	fout.close();
      }

      // Otherwise we output all the images

      else {
	
	seriesItr = seriesUID.begin();
	seriesEnd = seriesUID.end();
	
	int i = 0;
	while( seriesItr != seriesEnd ) {
	  
	  std::fstream fout;

	  fileOutput = SeriesOutputFilename( fout,
					     nameGenerator->GetFileNames( *seriesItr )[0],
					     args.fileOutputStem, args.flgVerbose,
					     tagModalityValue );

	  std::cout << i << ": Filename: " << fileOutput << std::endl;
	  
	  if ( WriteSeriesAsVolume( fout,
				    args.flgVerbose, 
				    args.flgDebug, 
				    args.orientation,
				    *seriesItr, 
				    args.fileNameFilter,
				    args.fileOutputStem, 
				    args.fileOutputSuffix, 
				    fileOutput, 
				    nameGenerator, 
				    reader,
				    tagModalityValue ) )
	    i++;
	  
	  fout.close();
	  seriesItr++;
	}
      }
    }
  catch (itk::ExceptionObject &ex)
    {
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
  
  return EXIT_SUCCESS;
}

