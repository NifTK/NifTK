/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include <iostream>
#include <fstream.h>
#include <cstdlib>
#include <vector>
#include <itksys/SystemTools.hxx>

/**
 * \file niftkFixIncludeGuards.cxx
 * \brief Will scan a header file for 2 lines representing an include guard,
 * and re-write the file to have a standardised include guard of
 *
 * #ifndef <filename>_<extension>
 * #define <filename>_<extension>
 *
 * This can then be used to run on the whole code-base.
 */
int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Usage: niftkFixIncludeGuards filename" << std::endl;
    return EXIT_FAILURE;
  }

  std::string fileExtension = itksys::SystemTools::GetFilenameLastExtension(argv[1]);
  std::string fileName = itksys::SystemTools::GetFilenameWithoutLastExtension(argv[1]);

  if (fileExtension.size() == 0)
  {
    std::cerr << "Failed to read file extension for " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }

  if (fileName.size() == 0)
  {
    std::cerr << "Failed to read file name for " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<std::string> completeFileByLine;
  std::string singleLine;

  // Read whole file in a line at a time into a buffer.
  ifstream myfile(argv[1]);
  while (std::getline(myfile, singleLine))
  {
    completeFileByLine.push_back(singleLine);
  }

  // Check we have file.
  if (completeFileByLine.size() < 2)
  {
    std::cerr << "File has only " << completeFileByLine.size() << " lines" << std::endl;
    return EXIT_FAILURE;
  }

  // Scan for 2 consecutive lines, each looking like an include guard.
  for (unsigned int i = 0; i < completeFileByLine.size()-1; ++i)
  {
    std::string thisLine = completeFileByLine[i];
    std::string nextLine = completeFileByLine[i+1];

    if (thisLine.substr(0, 7) == std::string("#ifndef")
      && nextLine.substr(0, 7) == std::string("#define")
      )
    {
      std::string thisGuard = thisLine.substr(7);
      std::string nextGuard = nextLine.substr(7);

      int firstSpace = thisGuard.find_first_of(' ');
      int nextSpace = nextGuard.find_first_of(' ');

      if (thisGuard.substr(0, firstSpace) == nextGuard.substr(0, nextSpace))
      {
        std::string guard = fileName + std::string("_") + fileExtension.substr(1);
        completeFileByLine[i] = std::string("#ifndef ") + guard;
        completeFileByLine[i+1] = std::string("#define ") + guard;
        break;
      }
    }
  }

  // Print file to overwrite file.
  ofstream outfile(argv[1]);
  for (unsigned int i = 0; i < completeFileByLine.size(); ++i)
  {
    outfile << completeFileByLine[i] << std::endl;
  }

  return EXIT_SUCCESS;
}
