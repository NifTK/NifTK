/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNVidiaSDIDataType_h
#define niftkNVidiaSDIDataType_h

#include <niftkIGIDataType.h>

namespace niftk
{

/**
 * \class NVidiaSDIDataType
 * \brief Class to represent video frame data from NVidia SDI, to integrate within the niftkIGI framework.
 */
class NVidiaSDIDataType : public IGIDataType
{
public:

  typedef unsigned long long NVidiaSDITimeType;

  virtual ~NVidiaSDIDataType();
  NVidiaSDIDataType(unsigned int, unsigned int, NVidiaSDITimeType);
  NVidiaSDIDataType(const NVidiaSDIDataType&);
  NVidiaSDIDataType(const NVidiaSDIDataType&&);
  NVidiaSDIDataType& operator=(const NVidiaSDIDataType&);
  NVidiaSDIDataType& operator=(const NVidiaSDIDataType&&);

  unsigned int GetSequenceNumber() const;
  unsigned int GetCookie() const;

  virtual void Clone(const IGIDataType&) override;

private:

  // Used internally to make sure this data item comes from a valid
  // capture session. Otherwise what could happen is that when signal drops out (e.g. due to
  // interference on the wire) the old capture context is destroyed and a new one is created 
  // (fairly quickly) but any in-flight NVidiaSDIDataType hanging around in the IGIDataSourceManager
  // might still reference the previous one.
  unsigned int       m_MagicCookie;

  // SDI sequence number. Starts counting at 1 and increases for every set of captured images.
  unsigned int       m_SequenceNumber;

  // The SDI card keeps a time stamp for each frame coming out of the wire.
  // This is in some arbitrary unit (nanoseconds?) in reference to some arbitrary clock.
  NVidiaSDITimeType  m_GpuArrivalTime;
};

} // end namespace

#endif
