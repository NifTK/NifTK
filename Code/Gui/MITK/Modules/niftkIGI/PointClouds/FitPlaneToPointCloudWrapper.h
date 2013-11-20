/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef FitPlaneToPointCloudWrapper_h
#define FitPlaneToPointCloudWrapper_h


#include "niftkIGIExports.h"
#include <mitkCommon.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkObjectFactoryBase.h>


namespace niftk
{


class NIFTKIGI_EXPORT FitPlaneToPointCloudWrapper : public itk::Object
{
public:
  mitkClassMacro(FitPlaneToPointCloudWrapper, itk::Object);
  itkNewMacro(FitPlaneToPointCloudWrapper);


protected:
  /** Not implemented */
  FitPlaneToPointCloudWrapper();
  /** Not implemented */
  virtual ~FitPlaneToPointCloudWrapper();

  /** Not implemented */
  FitPlaneToPointCloudWrapper(const FitPlaneToPointCloudWrapper&);
  /** Not implemented */
  FitPlaneToPointCloudWrapper& operator=(const FitPlaneToPointCloudWrapper&);
};


} // namespace

#endif // FitPlaneToPointCloudWrapper
