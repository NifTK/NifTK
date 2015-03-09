/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPNMWriterService_h
#define niftkPNMWriterService_h

#include <mitkAbstractFileWriter.h>

namespace niftk
{

/**
 * Writes PNM format images to disk
 * @ingroup Process
 */
class PNMWriterService : public mitk::AbstractFileWriter
{
public:

  PNMWriterService();
  virtual ~PNMWriterService();

  using mitk::AbstractFileWriter::Write;
  virtual void Write();

private:

  PNMWriterService(const PNMWriterService & other);
  virtual PNMWriterService * Clone() const;
};

} // end of namespace mitk

#endif
