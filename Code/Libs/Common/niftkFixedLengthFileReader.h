/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkFixedLengthFileReader_h
#define niftkFixedLengthFileReader_h

#include <vector>
#include <fstream>

namespace niftk
{

/**
 * @class FixedLengthFileReader
 * @brief Reads a fixed number S of type T from a file.
 *
 * This class provided to provide consistent file access for
 * reading files of a common type. All errors are reported as
 * niftk::IOException
 */
template<typename T, size_t S>
class FixedLengthFileReader
{
public:

  /**
   * @brief Constructor reads the data.
   * @param fileName path to file
   * @param strict if true the file must contain the exact number of values,
   * if false, will read the first S values out of a potentially larger file.
   */
  FixedLengthFileReader(const std::string& fileName, const bool& strict=true);
  virtual ~FixedLengthFileReader();

  /**
   * @brief retrieves a copy of the data.
   */
  std::vector<T> GetData() const;

private:

  FixedLengthFileReader(const FixedLengthFileReader&); // purposely not implemented
  void operator=(const FixedLengthFileReader&); // purposely not implemented

  /**
   * @brief Closes the file, as does the destructor, by calling this.
   */
  void CloseFile();

  std::ifstream  m_InputStream;
  std::vector<T> m_Data;
  std::string    m_FileName;

}; // end class

} // end namespace

#include "niftkFixedLengthFileReader.txx"

#endif
