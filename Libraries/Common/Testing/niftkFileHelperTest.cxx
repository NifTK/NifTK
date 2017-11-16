/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <niftkFileHelper.h>
#include <niftkEnvironmentHelper.h>
#include <niftkConversionUtils.h>
#include "Exceptions/niftkIOException.h"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

/**
 * \file FileHelperTest.cxx
 * \brief Defines unit tests for various file utilities.
 */

//-----------------------------------------------------------------------------
int TestCheckDirectoryEmptyInput()
{
  try {
    niftk::DirectoryExists("");
    std::cerr << "The method niftk::DirectoryExists("") should throw an exception if the input argument is empty." << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception&)
    {
      return EXIT_SUCCESS;
    }
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestCheckDirectoryNotADirectory()
{
  // Should return false
  if(niftk::DirectoryExists("blah"))
  {
    std::cerr << "The method niftk::DirectoryExists should return false for an invalid directory." << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestValidDirectory()
{
  fs::path curr_path = fs::current_path();
  if(!(fs::is_directory( curr_path )))
  {
    std::cerr << "The test is technically a test for boost, but it testing that fs::current_path() does return a directory." << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestMatchingPrefixAndExtension()
{
  if (! ( niftk::FilenameHasPrefixAndExtension("test1.txt", "", "txt")))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return true if the prefix is empty, as this is valid, and the file extension matches." << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! (niftk::FilenameHasPrefixAndExtension("test1.txt", "t", "txt")))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return true if the prefix is non empty and the file extension matches." << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! (niftk::FilenameHasPrefixAndExtension("test1.txt", "t", "xt")))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return true if the extension is not the bit after the last dot." << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! (niftk::FilenameHasPrefixAndExtension("test1.txt", "t", ".txt")) )
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return true if the specified extension contains the dot." << std::endl;
    return EXIT_FAILURE;
  }

  if ( niftk::FilenameHasPrefixAndExtension("test1.txt", "f", "txt") )
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return false if the specified prefix is not empty and is not contained at the start of the file name." << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! ( niftk::FilenameHasPrefixAndExtension("1.", "", "") ) )
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return true if the extension and prefix is not specified" << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! (niftk::FilenameHasPrefixAndExtension("test.extrinsic.xml", "", ".extrinsic.xml")))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return true if the extension has more than one full stop" << std::endl;
    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestMatchingLibraryFileName()
{
  if (!niftk::FilenameMatches("test1.txt", "", "test1", "txt"))
  {
    std::cerr << "The method niftk::FilenameMatches should return true if the prefix is empty (as this is valid), and the rest of the filename and extension matches." << std::endl;
    return EXIT_FAILURE;
  }

  if (!niftk::FilenameMatches("test1..txt", "t", "est1.", "txt"))
  {
    std::cerr << "The method niftk::FilenameMatches should return true in the case of double dots. The file extension is the bit after the last dot. The prefix here is t. And the middle section can itself contain dots." << std::endl;
    return EXIT_FAILURE;
  }

  if (niftk::FilenameMatches("test1.txt", "t", "est1.", "txt"))
  {
    std::cerr << "The method niftk::FilenameMatches should return false if the middle section contains the only dot, as that dot is the separator for the file extension." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestFileSeparator()
{

#if (defined(WIN32) || defined(_WIN32))
  if (!(niftk::GetFileSeparator() == "\\" ))
  {
    std::cerr << "The method niftk::GetFileSeparator() should return \\ on Windows" << std::endl;
    return EXIT_FAILURE;
  }
#else
  if (!(niftk::GetFileSeparator() == "/"))
  {
    std::cerr << "The method niftk::GetFileSeparator() should return / on Linux / Mac" << std::endl;
    return EXIT_FAILURE;
  }
#endif
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestConcatenateToPath()
{
#if (defined(WIN32) || defined(_WIN32))
  if (!(niftk::ConcatenatePath("a", "b") == "a\\b" ))
  {
    std::cerr << "The method niftk::ConcatenatePath should concatenate b onto a, and add the file separator." << std::endl;
    return EXIT_FAILURE;
  }
  if (!(niftk::ConcatenatePath("a", "") == "a" ))
  {
    std::cerr << "The method niftk::ConcatenatePath should not concatenate a file separator when the path is specified, but the filename is blank." << std::endl;
    return EXIT_FAILURE;
  }
  if (!(niftk::ConcatenatePath("", "") == "" ))
  {
    std::cerr << "If both path and file name are blank, there is no point adding a separator." << std::endl;
    return EXIT_FAILURE;
  }
  if (!(niftk::ConcatenatePath("", "b") == "b" ))
  {
    std::cerr << "If the file name is not blank, but the path is, then the output should just be the filename." << std::endl;
    return EXIT_FAILURE;
  }
#else
  if (!(std::string("a/b") == niftk::ConcatenatePath("a", "b")))
  {
    std::cerr << "The method niftk::ConcatenatePath should concatenate b onto a, and add the file separator." << std::endl;
    return EXIT_FAILURE;
  }
  if (!(std::string("a") == niftk::ConcatenatePath("a", "")))
  {
    std::cerr << "The method niftk::ConcatenatePath should not concatenate a file separator when the path is specified, but the filename is blank." << std::endl;
    return EXIT_FAILURE;
  }
  if (!(std::string("") == niftk::ConcatenatePath("", "")))
  {
    std::cerr << "If both path and file name are blank, there is no point adding a separator." << std::endl;
    return EXIT_FAILURE;
  }
  if (!(std::string("b") == niftk::ConcatenatePath("", "b")))
  {
    std::cerr << "If the file name is not blank, but the path is, then the output should just be the filename." << std::endl;
    return EXIT_FAILURE;
  }
#endif
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestExtractImageFileSuffix()
{
  std::string ostr;
  if (!((ostr=niftk::ExtractImageFileSuffix("")) == "" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "an empty string if the input is empty [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix("file.jpg.name.gz")) == "" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "an empty string if the image suffix is not at the end [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix("filename")) == "" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "an empty string if no image suffix is present (a) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix("1.2.34.556.7")) == "" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "an empty string if no image suffix is present (b) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix("somefile.bmp")) == ".bmp" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "the image suffix (a) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix("somefile.nii.gz")) == ".nii.gz" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "the image suffix (b) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix("somefile.dcm")) == ".dcm" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "the image suffix (c) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix("somefile.gipl")) == ".gipl" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "the image suffix (d) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix("1.dicom")) == ".dicom" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "the image suffix (e) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ExtractImageFileSuffix(".TIFF")) == ".TIFF" ))
  {
    std::cerr << "The method niftk::ExtractImageFileSuffix should return "
              << "the image suffix (f) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestAddStringToImageFileSuffix()
{
  std::string ostr;
  if (!((ostr=niftk::AddStringToImageFileSuffix("", "")) == "" ))
  {
    std::cerr << "The method niftk::AddStringToImageFileSuffix should "
              << "return an empty string if the input strings are empty [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::AddStringToImageFileSuffix("somefile.png", ""))
        == "somefile.png" ))
  {
    std::cerr << "The method niftk::AddStringToImageFileSuffix should "
              << "not change the file name if the string to be "
              << "added is empty [incorrect: " << ostr << "]." << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::AddStringToImageFileSuffix(".png", "somefile"))
        == "somefile.png" ))
  {
    std::cerr << "The method niftk::AddStringToImageFileSuffix should "
              << "return the concatenated file name with text added before "
              << "the suffix (a) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::AddStringToImageFileSuffix("somefile.img", "_add"))
        == "somefile_add.img" ))
  {
    std::cerr << "The method niftk::AddStringToImageFileSuffix should "
              << "return the concatenated file name with text added before "
              << "the suffix (b) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::AddStringToImageFileSuffix("somefile.abc", "_add"))
        == "somefile.abc_add" ))
  {
    std::cerr << "The method niftk::AddStringToImageFileSuffix should "
              << "not change the suffix of a non-image file (a) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::AddStringToImageFileSuffix("somefile.doc", "_add"))
        == "somefile.doc_add" ))
  {
    std::cerr << "The method niftk::AddStringToImageFileSuffix should "
              << "not change the suffix of a non-image file (b) [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::AddStringToImageFileSuffix("somefile.ima.gz", "_add"))
        == "somefile_add.ima.gz" ))
  {
    std::cerr << "The method niftk::AddStringToImageFileSuffix should "
              << "include the compression extension '.gz' in the suffix [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestModifyImageFileSuffix()
{
  std::string ostr;
  if (!((ostr=niftk::ModifyImageFileSuffix("", "")) == "" ))
  {
    std::cerr << "The method niftk::ModifyImageFileSuffix should "
              << "return an empty string if the inputs are empty [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ModifyImageFileSuffix("somefile.gz", "_new_"))
             == "somefile.gz_new_" ))
  {
    std::cerr << "The method niftk::ModifyImageFileSuffix should "
              << "append text to the file name if no image suffix is "
              << "found [incorrect: " << ostr << "]." << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ModifyImageFileSuffix("somefile.JPG", ".replace"))
             == "somefile.replace" ))
  {
    std::cerr << "The method niftk::ModifyImageFileSuffix should "
              << "replace the image suffix with the new string [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ModifyImageFileSuffix("somefile.tif.zip", ".replace"))
             == "somefile.replace" ))
  {
    std::cerr << "The method niftk::ModifyImageFileSuffix should "
              << "replace the image suffix and '.zip' extension"
              << " with the new string [incorrect: " << ostr << "]." << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestModifyFileSuffix()
{
  std::string ostr;
  if (!((ostr=niftk::ModifyFileSuffix("", "")) == "" ))
  {
    std::cerr << "The method niftk::ModifyFileSuffix should "
              << "return an empty string if the inputs are empty [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ModifyFileSuffix("file.txt", "")) == "file" ))
  {
    std::cerr << "The method niftk::ModifyFileSuffix should remove the suffix "
              << "if the input suffix has zero length [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ModifyFileSuffix("file.txt", ".txt")) == "file.txt" ))
  {
    std::cerr << "The method niftk::ModifyFileSuffix should return the input filename "
              << "if the input suffix is identical [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  else if (!((ostr=niftk::ModifyFileSuffix("file.txt", "_hello.txt"))
             == "file_hello.txt" ))
  {
    std::cerr << "The method niftk::ModifyFileSuffix should return the input filename "
              << "with the suffix replaced by the new string [incorrect: " << ostr << "]."
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestFindVideoData(std::string directory)
{
  std::vector<std::string> videoFiles = niftk::FindVideoData(directory);
  if ( videoFiles.size() != 4 )
  {
    std::cerr << "The method niftk::FindVideoData should "
              << "return 4 files for directory " << directory << ", not " << videoFiles.size()
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestFindVideoFile(std::string directory)
{
  std::string videoFile = niftk::FindVideoFile(directory, "capture-2013_7_26-16_0_36");
  std::string rightFile = directory + niftk::GetFileSeparator() + "QmitkIGINVidiaDataSource_5" + niftk::GetFileSeparator() + "capture-2013_7_26-16_0_36.264";
  if ( videoFile != rightFile )
  {
    std::cerr << "The method niftk::FindVideoFile should "
              << "find " << rightFile << " in the directory " << directory
              << ", not "<< videoFile
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestEmptyFile(std::string file)
{
  if ( ! niftk::FileIsEmpty ( file ) )
  {
    std::cerr << "The method niftk::FileIsEmpty should return true "
              << "for empty file " << file
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
int TestNotEmptyFile(std::string file)
{
  if ( niftk::FileIsEmpty ( file ) )
  {
    std::cerr << "The method niftk::FileIsEmpty should return false "
              << "for non empty file " << file
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestCreateAndDeleteUniqueFile()
{
  std::string filename;
  try
  {
    filename = niftk::CreateUniqueTempFileName("video",".avi");
  }
  catch (niftk::IOException e)
  {
    std::cerr << "The method niftk::CreateUniqueTempFileName did not return a "
              << "unique file name " << filename
              << " : " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! niftk::FileIsRegular ( filename ) )
  {
    std::cerr << "The method niftk::CreateUniqueTempFileName did not return a "
              << "writeable file name. " << filename << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! niftk::FileIsEmpty ( filename ) )
  {
    std::cerr << "The method niftk::CreateUniqueTempFileName did not return an "
              << "empty file. " << filename << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! niftk::FileDelete ( filename ) )
  {
    std::cerr << "The method niftk::FileDelete did not successfully "
              << "delete empty file. " << filename << std::endl;
    return EXIT_FAILURE;
  }

  if ( niftk::FileIsRegular ( filename ) )
  {
    std::cerr << "The method niftk::FileDelete did not delete "
              << filename << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestFileSize(std::string file, int minExpectedSize, int maxExpectedSize )
{
  if ( ! ( ( niftk::FileSize ( file ) >= minExpectedSize ) && ( niftk::FileSize ( file ) <= maxExpectedSize ) ) )
  {
    std::cerr << "The method niftk::FileSize should return "
              << minExpectedSize << " to " << maxExpectedSize
              << " for file " << file << ": not " << niftk::FileSize ( file )
              << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestFileIsRegular(std::string regularFile, std::string nonExistentFile, std::string directory )
{
  if ( ! niftk::FileIsRegular ( regularFile ) )
  {
    std::cerr << "The method niftk::FileIsRegular should return true for  "
              << regularFile
              << std::endl;
    return EXIT_FAILURE;
  }

  if ( niftk::FileIsRegular ( nonExistentFile ) )
  {
    std::cerr << "The method niftk::FileIsRegular should return false for  "
              << nonExistentFile
              << std::endl;
    return EXIT_FAILURE;
  }

  if ( niftk::FileIsRegular ( directory ) )
  {
    std::cerr << "The method niftk::FileIsRegular should return false for  "
              << directory
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestFileExists(std::string regularFile, std::string nonExistentFile, std::string directory )
{
  if ( ! niftk::FileExists ( regularFile ) )
  {
    std::cerr << "The method niftk::FileExists should return true for  "
              << regularFile
              << std::endl;
    return EXIT_FAILURE;
  }

  if ( niftk::FileExists ( nonExistentFile ) )
  {
    std::cerr << "The method niftk::FileExists should return false for  "
              << nonExistentFile
              << std::endl;
    return EXIT_FAILURE;
  }

  if ( ! niftk::FileExists ( directory ) )
  {
    std::cerr << "The method niftk::FileExists should return true for  "
              << directory
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestGetTQRDFileHeader()
{
  unsigned int headerSize[] = {0,100,256,1024};
  std::string header;

  for ( unsigned int i = 0; i < sizeof(headerSize)/sizeof(headerSize[0]); ++i )
  {
    try
    {
      header = niftk::GetTQRDFileHeader ( headerSize[i] );
      if ( header.length() != headerSize[i] )
      {
        std::cerr << "Calling niftk::GetTQRDFileHeader with header size " << headerSize[i] << " failed." << std::endl;
        return EXIT_FAILURE;
      }
      else
      {
        std::cout << "Calling niftk::GetTQRDFileHeader with header size " << headerSize[i] << " OK." << std::endl;
      }
    }
    catch ( niftk::IOException e )
    {
      int strcmpret = strncmp( "Target header size", e.what(), 18);
      if ( ( strcmpret != 0 ) )
      {
        std::cerr << strcmpret << ": Calling niftk::GetTQRDFileHeader with header size " << headerSize[i] <<
          " got unknown exception: " << e.what() << std::endl ;
        return EXIT_FAILURE;
      }
      else
      {
        std::cout << "Calling niftk::GetTQRDFileHeader with header size " << headerSize[i] << " threw correct exception OK." << std::endl;
      }
    }
  }
  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestCheckTQRDFileHeader(std::string valid, std::string invalid, std::string garbage)
{
  std::ifstream ifs;

  ifs.open(valid, std::ios::binary | std::ios::in);

  unsigned int fileHeaderSize = 256;
  try
  {
    niftk::CheckTQRDFileHeader(ifs, fileHeaderSize);
    std::cout << "Calling niftk::CheckTQRDFileHeader for " << valid << " OK." << std::endl;
  }
  catch ( niftk::IOException e )
  {
    std::cerr << "Calling niftk::CheckTQRDFileHeader for " << valid << " failed." << std::endl;
    return EXIT_FAILURE;
  }
  ifs.close();

  ifs.open(invalid, std::ios::binary | std::ios::in);

  try
  {
    niftk::CheckTQRDFileHeader(ifs, fileHeaderSize);
    std::cerr << "Calling niftk::CheckTQRDFileHeader for " << invalid << " succeeded, but should have failed." << std::endl;
    return EXIT_FAILURE;
  }
  catch ( niftk::IOException e )
  {
    std::cerr << "Calling niftk::CheckTQRDFileHeader for " << invalid << " failed, as it should. OK." << std::endl;
  }
  ifs.close();

  ifs.open(garbage, std::ios::binary | std::ios::in);

  try
  {
    niftk::CheckTQRDFileHeader(ifs, fileHeaderSize);
    std::cerr << "Calling niftk::CheckTQRDFileHeader for " << garbage << " succeeded, but should have failed." << std::endl;
    return EXIT_FAILURE;
  }
  catch ( niftk::IOException e )
  {
    std::cerr << "Calling niftk::CheckTQRDFileHeader for " << garbage << " failed, as it should. OK." << std::endl;
  }

  fileHeaderSize = 0;
  try
  {
    niftk::CheckTQRDFileHeader(ifs, fileHeaderSize);
    std::cerr << "Calling niftk::CheckTQRDFileHeader with zero length header should fail." << std::endl;
    return EXIT_FAILURE;
  }
  catch ( niftk::IOException e )
  {
    std::cout << "Calling niftk::CheckTQRDFileHeader with zero length header failed, OK." << std::endl;
  }

  ifs.close();

  ifs.open(valid, std::ios::binary | std::ios::in);

  fileHeaderSize = 100;
  //This might not fail for larger values of fileHeaderSize - as long the length is long enough to get to the padding.
  try
  {
    niftk::CheckTQRDFileHeader(ifs, fileHeaderSize);
    std::cout << "Calling niftk::CheckTQRDFileHeader with short header length ("<< fileHeaderSize << ") should fail." << std::endl;
    return EXIT_FAILURE;
  }
  catch ( niftk::IOException e )
  {
    std::cerr << "Calling niftk::CheckTQRDFileHeader with short header length ("<< fileHeaderSize << ") fail. OK." << std::endl;
  }
  ifs.close();

  ifs.open(valid, std::ios::binary | std::ios::in);

  fileHeaderSize = 257;
  try
  {
    niftk::CheckTQRDFileHeader(ifs, fileHeaderSize);
    std::cout << "Calling niftk::CheckTQRDFileHeader with long header length ("<< fileHeaderSize << ") should fail." << std::endl;
    return EXIT_FAILURE;
  }
  catch ( niftk::IOException e )
  {
    std::cerr << "Calling niftk::CheckTQRDFileHeader with long header length ("<< fileHeaderSize << ") should fail. OK." << std::endl;
  }

  ifs.close();

  return EXIT_SUCCESS;
}

//-----------------------------------------------------------------------------
int TestCreateUniqueString()
{
  std::string filename;
  try
  {
    unsigned int targetLength[4] = {6,6,0,3};
    std::string laststring;
    for ( unsigned int i = 0 ; i < 4 ; ++i )
    {
      unsigned int seed = time(NULL) + i;
      filename = niftk::CreateUniqueString(targetLength[i], seed);
      if ( filename.length() != targetLength[i] )
      {
        std::cerr << "The method niftk::CreateUniqueString returned string of wrong length: "
                  << filename.length() << " -ne " << targetLength[i] << std::endl;
        return EXIT_FAILURE;
      }
      else
      {
        std::cout << "niftk::CreateUniqueString length " << targetLength[i] << " OK: " << filename << std::endl;
        if ( i == 1 )
        {
          if ( laststring == filename )
          {
            std::cerr << "The method niftk::CreateUniqueString returned non unique strings "
                  << filename << " == " << laststring << std::endl;
            return EXIT_FAILURE;
          }
        }
      }
      laststring = filename;
    }
  }
  catch (std::exception e)
  {
    std::cerr << "The method niftk::CreateUniqueString threw an exception "
              << " : " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/**
 * \brief Basic test harness for FileHelper.h
 */
int niftkFileHelperTest(int argc, char * argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage   :niftkFileHelperTest testNumber" << std::endl;
      return 1;
    }

  int testNumber = atoi(argv[1]);

  if (testNumber == 1)
  {
    return TestCheckDirectoryEmptyInput();
  }
  else if (testNumber == 2)
  {
    return TestCheckDirectoryNotADirectory();
  }
  else if (testNumber == 3)
  {
    return TestValidDirectory();
  }
  else if (testNumber == 4)
  {
    return TestMatchingPrefixAndExtension();
  }
  else if (testNumber == 5)
  {
    return TestMatchingLibraryFileName();
  }
  else if (testNumber == 6)
  {
    return TestFileSeparator();
  }
  else if (testNumber == 7)
  {
    return TestConcatenateToPath();
  }
  else if (testNumber == 8)
  {
    return TestExtractImageFileSuffix();
  }
  else if (testNumber == 9)
  {
    return TestAddStringToImageFileSuffix();
  }
  else if (testNumber == 10)
  {
    return TestModifyImageFileSuffix();
  }
  else if (testNumber == 11)
  {
    return TestModifyFileSuffix();
  }
   else if (testNumber == 12)
  {
    return TestFindVideoData(argv[2]);
  }
   else if (testNumber == 13)
  {
    return TestFindVideoFile(argv[2]);
  }
   else if (testNumber == 14)
  {
    return TestEmptyFile(argv[2]);
  }
   else if (testNumber == 15)
  {
    return TestNotEmptyFile(argv[2]);
  }
   else if (testNumber == 16)
  {
    return TestCreateAndDeleteUniqueFile();
  }
   else if (testNumber == 17)
  {
    return TestFileSize(argv[2], atoi(argv[3]), atoi(argv[4]));
  }
   else if (testNumber == 18)
  {
    return TestFileIsRegular(argv[2], argv[3], argv[4]);
  }
   else if (testNumber == 19)
  {
    return TestFileExists(argv[2], argv[3], argv[4]);
  }
   else if (testNumber == 20)
  {
    return TestGetTQRDFileHeader();
  }
   else if (testNumber == 21)
  {
    return TestCheckTQRDFileHeader(argv[2], argv[3], argv[4]);
  }
   else if (testNumber == 22)
  {
    return TestCreateUniqueString();
  }


  else
  {
    return EXIT_FAILURE;
  }
}

