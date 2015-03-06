/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkPNMWriter_h
#define __mitkPNMWriter_h

#include <mitkAbstractFileWriter.h>
#include <vtkPolyDataWriter.h>

namespace mitk
{

/**
 * Writes PNM format images to disk
 * @ingroup Process
 */
class PNMWriter : public mitk::AbstractFileWriter
{
public:
  PNMWriter();
  PNMWriter(const PNMWriter & other);
  virtual PNMWriter * Clone() const;
  virtual ~PNMWriter();

  using mitk::AbstractFileWriter::Write;
  virtual void Write();
};


} // end of namespace mitk

#endif //__mitkPNMWriter_h
