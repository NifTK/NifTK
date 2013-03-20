/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKIGINVIDIADATATYPE_H
#define MITKIGINVIDIADATATYPE_H

#include "niftkNVidiaExports.h"
#include "mitkIGIDataType.h"

namespace mitk
{

/**
 * \class IGINVidiaDataType
 * \brief Class to represent video frame data from NVidia SDI, to integrate within the niftkIGI framework.
 */
class NIFTKNVIDIA_EXPORT IGINVidiaDataType : public IGIDataType
{
public:

  mitkClassMacro(IGINVidiaDataType, IGIDataType);
  itkNewMacro(IGINVidiaDataType);

  // FIXME: i wish these were constructor parameters
  void set_values(unsigned int cookie, unsigned int sn, unsigned __int64 gputime);


protected:

  IGINVidiaDataType(); // Purposefully hidden.
  virtual ~IGINVidiaDataType(); // Purposefully hidden.

  IGINVidiaDataType(const IGINVidiaDataType&); // Purposefully not implemented.
  IGINVidiaDataType& operator=(const IGINVidiaDataType&); // Purposefully not implemented.

private:
  // Used internally by QmitkIGINVidiaDataSource to make sure this data item comes from a valid
  // capture session. Otherwise what could happen is that when signal drops out (e.g. due to
  // interference on the wire) the old capture context is destroyed and a new one is created 
  // (fairly quickly) but any in-flight IGINVidiaDataType hanging around in the QmitkIGIDataSourceManager
  // might still reference the previous one.
  unsigned int    magic_cookie;

  // SDI sequence number. Starts counting at 1 and increases for every set of captured images.
  unsigned int    sequence_number;

  // The SDI card keeps a time stamp for each frame coming out of the wire.
  // This is in some arbitrary unit (nanoseconds?) in reference to some arbitrary clock.
  unsigned __int64  gpu_arrival_time;
};

} // end namespace

#endif // MITKIGINVIDIADATATYPE_H
