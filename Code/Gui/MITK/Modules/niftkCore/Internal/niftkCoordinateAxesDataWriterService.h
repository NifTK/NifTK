/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCoordinateAxesDataWriterService_h
#define mitkCoordinateAxesDataWriterService_h

#include <mitkAbstractFileWriter.h>
#include <mitkCoordinateAxesData.h>

namespace niftk
{

/**
 * @internal
 * @brief Saves a mitk::CoordinateAxesData into a 4x4 Matrix text file.
 */
class CoordinateAxesDataWriterService : public mitk::AbstractFileWriter
{
public:

  CoordinateAxesDataWriterService();
  virtual ~CoordinateAxesDataWriterService();

  using AbstractFileWriter::Write;
  virtual void Write();

private:

  CoordinateAxesDataWriterService(const CoordinateAxesDataWriterService& other);
  virtual CoordinateAxesDataWriterService* Clone() const;
};

} // end namespace

#endif
