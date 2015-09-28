/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCoordinateAxesDataReaderService_h
#define niftkCoordinateAxesDataReaderService_h

#include <mitkAbstractFileReader.h>

namespace niftk
{

/**
* @class CoordinateAxesDataReaderService
* @brief Loads a 4x4 Matrix into a mitk::CoordinateAxesData.
* @internal
*/
class CoordinateAxesDataReaderService: public mitk::AbstractFileReader
{
public:

  CoordinateAxesDataReaderService();
  virtual ~CoordinateAxesDataReaderService();

  virtual std::vector< itk::SmartPointer<mitk::BaseData> > Read();

private:

  CoordinateAxesDataReaderService(const CoordinateAxesDataReaderService& other);
  virtual CoordinateAxesDataReaderService* Clone() const;
};

} // end namespace

#endif
