/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "FileHelper.h"
#include "stdlib.h"
#include "EnvironmentHelper.h"
#include <iostream>
#include "ConversionUtils.h"

namespace fs = boost::filesystem;

int testCheckDirectoryEmptyInput()
{
  // Should throw exception
  try {
    niftk::DirectoryExists("");
    return EXIT_FAILURE;
  } catch (std::exception&)
    {
      return EXIT_SUCCESS;
    }
  return EXIT_SUCCESS;
}

int testCheckDirectoryNotADirectory()
{
  // Should return false
  if(niftk::DirectoryExists("blah")) throw std::exception();
  return EXIT_SUCCESS;
}

int testValidDirectory()
{
  fs::path curr_path = fs::current_path();
  if(!(fs::is_directory( curr_path ))) throw std::exception();
  return EXIT_SUCCESS;
}

int testMatchingPrefixAndExtension()
{
  if (!(niftk::FilenameHasPrefixAndExtension("test1.txt", "", "txt"))) throw std::exception();
  if (!(niftk::FilenameHasPrefixAndExtension("test1.txt", "t", "txt"))) throw std::exception();
  if (niftk::FilenameHasPrefixAndExtension("test1.txt", "t", "xt")) throw std::exception();   // incomplete extension
  if (niftk::FilenameHasPrefixAndExtension("test1.txt", "t", ".txt")) throw std::exception(); // extension shouldnt contain dot.  
  if (niftk::FilenameHasPrefixAndExtension("test1.txt", "f", "txt")) throw std::exception();  // incorrect start
  if (!(niftk::FilenameHasPrefixAndExtension("1.", "", ""))) throw std::exception();          // this should work.
  return EXIT_SUCCESS;
}

int testMatchingLibraryFileName()
{
  if (!(niftk::FilenameMatches("test1.txt", "", "test1", "txt"))) throw std::exception();
  if (!(niftk::FilenameMatches("test1..txt", "t", "est1.", "txt"))) throw std::exception();  // ie filename ends in dot.
  if (niftk::FilenameMatches("test1.txt", "t", "est1.", "txt")) throw std::exception();
  return EXIT_SUCCESS;
}

int testFileSeparator()
{

#if (defined(WIN32) || defined(_WIN32))
  if (!(niftk::GetFileSeparator() == "\\" ))  throw std::exception();
#else
  if (!(niftk::GetFileSeparator() == "/")) throw std::exception();
#endif
  return EXIT_SUCCESS;
}

int testConcatenateToPath()
{
#if (defined(WIN32) || defined(_WIN32))
  if (!(niftk::ConcatenatePath("a", "b") == "a\\b" )) throw std::exception();
  if (!(niftk::ConcatenatePath("a", "") == "a\\" )) throw std::exception();
  if (!(niftk::ConcatenatePath("", "") == "\\" )) throw std::exception();
  if (!(niftk::ConcatenatePath("", "b") == "\\b" )) throw std::exception();
#else
  if (!(std::string("a/b") == niftk::ConcatenatePath("a", "b"))) throw std::exception();
  if (!(std::string("a/") == niftk::ConcatenatePath("a", ""))) throw std::exception();
  if (!(std::string( "/") == niftk::ConcatenatePath("", ""))) throw std::exception();
  if (!(std::string("/b") == niftk::ConcatenatePath("", "b"))) throw std::exception();
#endif
  return EXIT_SUCCESS;
}

/**
 * Basic test harness for FileHelper.h
 */
int FileHelperTest(int argc, char * argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage   :FileHelperTest testNumber" << std::endl;
      return 1;
    }
  
  int testNumber = atoi(argv[1]);
  
  if (testNumber == 1)
    {
      return testCheckDirectoryEmptyInput();
    }
  else if (testNumber == 2)
    {
      return testCheckDirectoryNotADirectory();
    }
  else if (testNumber == 3)
    {
      return testValidDirectory(); 
    }
  else if (testNumber == 4)
    {
      return testMatchingPrefixAndExtension();  
    }
  else if (testNumber == 5)
    {
      return testMatchingLibraryFileName();  
    }
  else if (testNumber == 6)
    {
      return testFileSeparator();  
    }
  else if (testNumber == 7)
    {
      return testConcatenateToPath();  
    }
  else
    {
      return EXIT_FAILURE;
    }
}

