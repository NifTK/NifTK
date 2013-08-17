/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCoordinateAxesDataWriter_h
#define mitkCoordinateAxesDataWriter_h

#include "niftkCoreExports.h"
#include <itkProcessObject.h>
#include <mitkFileWriterWithInformation.h>
#include <mitkCoordinateAxesData.h>

namespace mitk
{

/**
 * \class CoordinateAxesDataWriter
 * \brief Writes mitk::CoordinateAxesDataWriter to file.
 */
class NIFTKCORE_EXPORT CoordinateAxesDataWriter : public FileWriterWithInformation
{
public:

  mitkClassMacro( CoordinateAxesDataWriter, FileWriterWithInformation );
  itkNewMacro( Self );

  typedef mitk::CoordinateAxesData InputType;

  /**
   * \see mitk::FileWriter::SetFileName()
   */
  itkSetStringMacro( FileName );

  /**
   * \see mitk::FileWriter::GetFileName()
   */
  itkGetStringMacro( FileName );

  /**
   * \see mitk::FileWriter::SetFilePrefix()
   */
  itkSetStringMacro( FilePrefix );

  /**
   * \see mitk::FileWriter::GetFilePrefix()
   */
  itkGetStringMacro( FilePrefix );

  /**
   * \see mitk::FileWriter::SetFilePattern()
   */
  itkSetStringMacro( FilePattern );

  /**
   * \see mitk::FileWriter::GetFilePattern()
   */
  itkGetStringMacro( FilePattern );

  /**
   * \see mitk::FileWriter::GetSuccess()
   */
  itkGetMacro( Success, bool );

  /**
   * \see mitk::FileWriter::GetInput()
   */
  InputType* GetInput();

  /**
   * \see mitk::FileWriter::GetPossibleFileExtensions()
   * \brief Returns mitk::CoordinateAxesData::FILE_EXTENSION
   */
  std::vector<std::string> GetPossibleFileExtensions();

  const char * GetDefaultExtension();
  const char * GetDefaultFilename();
  const char * GetFileDialogPattern();

  void Write();
  void Update();
  bool CanWriteBaseDataType(BaseData::Pointer data);
  void DoWrite(BaseData::Pointer data);

protected:

  CoordinateAxesDataWriter(); // Purposefully hidden.
  virtual ~CoordinateAxesDataWriter(); // Purposefully hidden.

  virtual void GenerateData();

private:

  CoordinateAxesDataWriter(const CoordinateAxesDataWriter&); // Purposefully not implemented.
  CoordinateAxesDataWriter& operator=(const CoordinateAxesDataWriter&); // Purposefully not implemented.

  /**
   * Sets the input object for the filter.
   * \param input the mitk::CoordinateAxesData
   */
  void SetInputCoordinateAxesData( InputType* input );

  std::string m_FileName;
  std::string m_FilePrefix;
  std::string m_FilePattern;
  bool        m_Success;
};

} // end of namespace mitk

#endif // CoordinateAxesDataWriter_h
