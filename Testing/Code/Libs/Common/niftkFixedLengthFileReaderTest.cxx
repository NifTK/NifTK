/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkFixedLengthFileReader.h>
#include <cstdlib>
#include <iostream>
#include <niftkIOException.h>

#define FILE_READER_EXCEPTION_TEST(number, type, size, filename, strict,errorMsg, expectedException) \
  try \
  { \
    niftk::FixedLengthFileReader<type, size> someReader(filename, strict); \
    std::cerr << errorMsg << std::endl; \
    return EXIT_FAILURE; \
  } \
  catch (const expectedException& e) \
  { \
  } \
  catch (const std::runtime_error& e) \
  { \
    std::cerr << "Caught runtime_error, when should be expectedException" << std::endl; \
    return EXIT_FAILURE;\
  } \
  std::cout << "Passed test " << number << std::endl;

/**
 * \brief Basic test harness for FileHelper.h
 */
int niftkFixedLengthFileReaderTest(int argc, char * argv[])
{
  if (argc < 1)
  {
    std::cerr << "Usage: niftkFixedLengthFileReaderTest file" << std::endl;
    return EXIT_FAILURE;
  }

  FILE_READER_EXCEPTION_TEST(1, double, 0, "whatever", false, "Should have thrown due to size being zero", niftk::IOException);
  FILE_READER_EXCEPTION_TEST(2, double, 1, "", false, "Should have thrown due to filename being empty", niftk::IOException);
  FILE_READER_EXCEPTION_TEST(3, double, 1, "nonsense", false, "Should have thrown due to filename not existing", niftk::IOException);
  FILE_READER_EXCEPTION_TEST(4, double, 15, "/Users/mattclarkson/Data/id.4x4", true, "Should have thrown due to strict mode not wanting files that are too long", niftk::IOException);

  niftk::FixedLengthFileReader<double, 16> readerFor4x4Matrix("/Users/mattclarkson/Data/id.4x4");
  std::vector<double> result = readerFor4x4Matrix.GetData();
  if (result.size() != 16)
  {
    std::cerr << "Expecting 16 doubles, but got " << result.size() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

