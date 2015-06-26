/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkICPRegServiceRAII_h
#define niftkICPRegServiceRAII_h

#include <niftkIGIServicesExports.h>
#include "niftkSurfaceRegServiceI.h"

#include <usServiceReference.h>
#include <usModuleContext.h>

#include <mitkDataNode.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
* \class ICPRegServiceRAII
* \brief RAII object to run ICP Surface Based Registration.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof. 
*/
class NIFTKIGISERVICES_EXPORT ICPRegServiceRAII : public SurfaceRegServiceI
{

public:

  /**
  * \brief Obtains service or throws mitk::Exception.
  * \param maxLandmarks The number of points to use in the ICP.
  * \param maxIterations The number of iterations to use in the ICP.
  */
  ICPRegServiceRAII(const int& maxLandmarks, const int& maxIterations);

  /**
  * \brief Releases service.
  */
  virtual ~ICPRegServiceRAII();

  /**
  * \brief Calls service to do ICP Surface Based Registration.
  * \see SurfaceRegServiceI::Register()
  */
  virtual double Register(const mitk::DataNode::Pointer fixedDataSet,
                          const mitk::DataNode::Pointer movingDataSet,
                          vtkMatrix4x4& matrix) const;

private:
  ICPRegServiceRAII(const ICPRegServiceRAII&); // deliberately not implemented
  ICPRegServiceRAII& operator=(const ICPRegServiceRAII&); // deliberately not implemented

  us::ModuleContext*                                     m_ModuleContext;
  std::vector<us::ServiceReference<SurfaceRegServiceI> > m_Refs;
  niftk::SurfaceRegServiceI*                             m_Service;
};

} // end namespace

#endif
