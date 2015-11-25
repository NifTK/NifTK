/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceServiceI_h
#define niftkIGIDataSourceServiceI_h

#include <niftkIGIServicesExports.h>
#include <mitkServiceInterface.h>

namespace niftk
{

/**
* \class IGIDataSourceServiceI
* \brief Interface for a IGI Data Source (e.g. video feed, ultrasound feed, tracker feed).
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*
* Note: Implementors of this interface must be thread-safe.
*/
class NIFTKIGISERVICES_EXPORT IGIDataSourceServiceI
{

public:

  virtual void StartCapturing() = 0;
  virtual void StopCapturing() = 0;
  virtual void StartRecording() = 0;
  virtual void StopRecording() = 0;
  virtual void SetLagInMilliseconds(const unsigned long long& milliseconds) = 0;
  virtual void SetRecordingLocation(const std::string& pathName) = 0;

protected:
  IGIDataSourceServiceI();
  virtual ~IGIDataSourceServiceI();

private:
  IGIDataSourceServiceI(const IGIDataSourceServiceI&); // deliberately not implemented
  IGIDataSourceServiceI& operator=(const IGIDataSourceServiceI&); // deliberately not implemented
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::IGIDataSourceServiceI, "uk.ac.ucl.cmic.IGIDataSourceServiceI");

#endif
