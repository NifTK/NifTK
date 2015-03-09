/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPNMReaderService_h
#define niftkPNMReaderService_h

#include <mitkAbstractFileReader.h>

namespace niftk
{

/**
 * @class PNMReaderService
 * @brief The PNMReaderService class
 */
class PNMReaderService : public mitk::AbstractFileReader
{
public:

  PNMReaderService();
  virtual ~PNMReaderService();

  using mitk::AbstractFileReader::Read;
  virtual std::vector<itk::SmartPointer<mitk::BaseData> > Read();

private:

  PNMReaderService(const PNMReaderService& other);
  virtual PNMReaderService * Clone() const;

  us::ServiceRegistration<mitk::IFileReader> m_ServiceReg;
};

} // namespace niftk

#endif
