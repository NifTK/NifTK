/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSurfaceRegServiceRAII_h
#define niftkSurfaceRegServiceRAII_h

#include <niftkIGIServicesExports.h>
#include "niftkSurfaceRegServiceI.h"

#include <usServiceReference.h>
#include <usModuleContext.h>

#include <mitkDataNode.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
* \class SurfaceRegServiceRAII
* \brief RAII object to run Surface Based Registration via a SurfaceRegServiceI implementation.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGISERVICES_EXPORT SurfaceRegServiceRAII : public SurfaceRegServiceI
{

public:

  /**
  * \brief Obtains service or throws mitk::Exception.
  */
  SurfaceRegServiceRAII(const std::string& method="ICP");

  /**
  * \brief Releases service.
  */
  virtual ~SurfaceRegServiceRAII();

  /**
  * \brief Calls service to do Point Based Registration.
  * \see SurfaceRegServiceI
  */
  virtual double SurfaceBasedRegistration(const mitk::DataNode::Pointer& fixedDataSet,
                                          const mitk::DataNode::Pointer& movingDataSet,
                                          vtkMatrix4x4& matrix) const;

private:
  SurfaceRegServiceRAII(const SurfaceRegServiceRAII&); // deliberately not implemented
  SurfaceRegServiceRAII& operator=(const SurfaceRegServiceRAII&); // deliberately not implemented

  us::ModuleContext*                                     m_ModuleContext;
  std::vector<us::ServiceReference<SurfaceRegServiceI> > m_Refs;
  niftk::SurfaceRegServiceI*                             m_Service;
};

} // end namespace

#endif
