/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCoordinateAxesDataReaderService_h
#define mitkCoordinateAxesDataReaderService_h

#include <mitkAbstractFileReader.h>
#include <mitkCoordinateAxesData.h>

namespace niftk
{

/**
 * @internal
 * @brief Loads a 4x4 Matrix into a mitk::CoordinateAxesData.
 */
class CoordinateAxesDataReaderService: public mitk::AbstractFileReader
{
public:

  CoordinateAxesDataReaderService();
  virtual ~CoordinateAxesDataReaderService();

  using AbstractFileReader::Read;
  virtual std::vector< itk::SmartPointer<mitk::BaseData> > Read();

private:

  CoordinateAxesDataReaderService(const CoordinateAxesDataReaderService& other);
  virtual CoordinateAxesDataReaderService* Clone() const;
};

} // end namespace

#endif
