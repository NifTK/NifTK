/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCoordinateAxesDataReader_h
#define mitkCoordinateAxesDataReader_h

#include "niftkCoreExports.h"
#include <mitkCommon.h>
#include <mitkFileReader.h>
#include <mitkCoordinateAxesData.h>

namespace mitk
{

/**
 * \class CoordinateAxesDataReader
 * \brief The reader for mitk::CoordinateAxesData
 */

class NIFTKCORE_EXPORT CoordinateAxesDataReader : public FileReader, public BaseProcess
{
public:

  mitkClassMacro( CoordinateAxesDataReader, BaseProcess );
  itkNewMacro(Self);

  typedef mitk::CoordinateAxesData OutputType;

  /**
   * \see mitk::FileReader::GetFileName()
   */
  virtual const char* GetFileName() const;

  /**
   * \see mitk::FileReader::SetFileName()
   */
  virtual void SetFileName(const char* aFileName);

  /**
   * \see mitk::FileReader::GetFilePrefix()
   */
  virtual const char* GetFilePrefix() const;

  /**
   * \see mitk::FileReader::SetFilePrefix()
   */
  virtual void SetFilePrefix(const char* aFilePrefix);

  /**
   * \see mitk::FileReader::GetFilePattern()
   */
  virtual const char* GetFilePattern() const;

  /**
   * \see mitk::FileReader::SetFilePattern()
   */
  virtual void SetFilePattern(const char* aFilePattern);

  /**
   * \see itk::ProcessObject::Update()
   */
  virtual void Update();

  /**
   * \brief Checks if the filename has the extension .4x4
   */
  static bool CanReadFile(const std::string filename,
                          const std::string filePrefix,
                          const std::string filePattern);

protected:

  CoordinateAxesDataReader(); // Purposefully hidden.
  virtual ~CoordinateAxesDataReader(); // Purposefully hidden.

  /**
   * \see itk::ProcessObject::GenerateData()
   */
  virtual void GenerateData();

  /**
   * \see itk::ProcessObject::GenerateOutputInformation()
   * \brief Loads the data using mitk::LoadVtkMatrix4x4FromFile(m_FileName);
   */
  virtual void GenerateOutputInformation();

private:

  CoordinateAxesDataReader(const CoordinateAxesDataReader&); // Purposefully not implemented.
  CoordinateAxesDataReader& operator=(const CoordinateAxesDataReader&); // Purposefully not implemented.

  std::string m_FileName;
  std::string m_FilePrefix;
  std::string m_FilePattern;

  OutputType::Pointer m_OutputCache;
};

} // namespace mitk

#endif // CoordinateAxesDataReader_h
