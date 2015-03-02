/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkFixedLengthFileReader.h"
#include <niftkIOException.h>
#include <niftkFileHelper.h>
#include <sstream>
#include <iostream>

namespace niftk
{

//-----------------------------------------------------------------------------
template<typename T, size_t S>
FixedLengthFileReader<T,S>::FixedLengthFileReader(const std::string& fileName, const bool& strict)
{
  if (S == 0)
  {
    throw niftk::IOException("Zero elements requested!");
  }

  if (fileName.size() == 0)
  {
    throw niftk::IOException("Empty file name supplied!");
  }

  if (!niftk::FileExists(fileName))
  {
    std::ostringstream oss;
    oss << "File '" << fileName << "' does not exist!";
    throw niftk::IOException(oss.str());
  }

  if (niftk::DirectoryExists(fileName))
  {
    std::ostringstream oss;
    oss << "File '" << fileName << "' is not a file, it is a directory!";
    throw niftk::IOException(oss.str());
  }

  m_InputStream.open(fileName.c_str());
  if (!m_InputStream.is_open())
  {
    std::ostringstream oss;
    oss << "Failed to open file '" << fileName << "'!";
    throw niftk::IOException(oss.str());
  }

  m_FileName = fileName;

  T value;
  m_Data.reserve(S);

  for (size_t i = 0; i < S; i++)
  {
    m_InputStream >> value;

    if (m_InputStream.bad() || m_InputStream.fail())
    {
      std::ostringstream oss;
      oss << "Failed to read element " << i << " from file '" << fileName << "'!";
      throw niftk::IOException(oss.str());
    }
    if (m_InputStream.eof())
    {
      std::ostringstream oss;
      oss << "Failed to read element " << i << " from file '" << fileName << "', due to end of file!";
      throw niftk::IOException(oss.str());
    }

    m_Data.push_back(value);
  }

  if (!m_InputStream.eof() && strict)
  {
    m_InputStream >> value;
    if (!m_InputStream.eof())
    {
      std::ostringstream oss;
      oss << "Reading " << S << " elements from file '" << fileName << "', and too many values are in the file!";
      throw niftk::IOException(oss.str());
    }
    m_InputStream.clear();
  }
  this->CloseFile();
}


//-----------------------------------------------------------------------------
template<typename T, size_t S>
FixedLengthFileReader<T,S>::~FixedLengthFileReader()
{
  try
  {
    this->CloseFile();
  }
  catch (const std::runtime_error& )
  {
    std::cerr << "Failed to close file '" << m_FileName << "', during ~FixedLengthFileReader()." << std::endl;
  }
}


//-----------------------------------------------------------------------------
template<typename T, size_t S>
typename std::vector<T> FixedLengthFileReader<T,S>::GetData() const
{
  return m_Data;
}


//-----------------------------------------------------------------------------
template<typename T, size_t S>
void FixedLengthFileReader<T,S>::CloseFile()
{
  if (m_InputStream.is_open())
  {
    m_InputStream.close();
    if (m_InputStream.fail())
    {
      std::ostringstream oss;
      oss << "Failed to close file '" << m_FileName << "'!";
      throw niftk::IOException(oss.str());
    }
  }
}

} // end namespace
