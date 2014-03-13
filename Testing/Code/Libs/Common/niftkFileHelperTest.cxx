/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <iostream>
#include <stdlib.h>
#include <niftkFileHelper.h>
#include <niftkEnvironmentHelper.h>
#include <niftkConversionUtils.h>

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
  if (!niftk::FilenameHasPrefixAndExtension("test1.txt", "", "txt"))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return true if the prefix is empty, as this is valid, and the file extension matches." << std::endl;
    return EXIT_FAILURE;
  }

  if (!niftk::FilenameHasPrefixAndExtension("test1.txt", "t", "txt"))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return true if the prefix is non empty and the file extension matches." << std::endl;
    return EXIT_FAILURE;
  }

  if (niftk::FilenameHasPrefixAndExtension("test1.txt", "t", "xt"))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return false if the extension is not the bit after the last dot." << std::endl;
    return EXIT_FAILURE;
  }

  if (niftk::FilenameHasPrefixAndExtension("test1.txt", "t", ".txt"))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return false if the specified extension contains the dot." << std::endl;
    return EXIT_FAILURE;
  }

  if (niftk::FilenameHasPrefixAndExtension("test1.txt", "f", "txt"))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return false if the specified prefix is not empty and is not contained at the start of the file name." << std::endl;
    return EXIT_FAILURE;
  }

  if (!(niftk::FilenameHasPrefixAndExtension("1.", "", "")))
  {
    std::cerr << "The method niftk::FilenameHasPrefixAndExtension should return false if the the extension is not specified, even if this matches the filename." << std::endl;
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
  else
  {
    return EXIT_FAILURE;
  }
}

