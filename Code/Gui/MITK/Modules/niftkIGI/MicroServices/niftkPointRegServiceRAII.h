/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPointRegServiceRAII_h
#define niftkPointRegServiceRAII_h

#include <niftkIGIExports.h>
#include "niftkPointRegServiceI.h"

#include <usServiceReference.h>
#include <usModuleContext.h>

#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
* \class RAII object to run Point Based Registration via a PointRegServiceI implementation.
*/
class NIFTKIGI_EXPORT PointRegServiceRAII : public PointRegServiceI
{

public:

  /**
  * @brief Obtains service or throws mitk::Exception.
  */
  PointRegServiceRAII(const std::string& method="SVD");

  /**
  * @brief Releases service.
  */
  virtual ~PointRegServiceRAII();

  /**
  * @brief Calls service to do Point Based Registration.
  * @see PointRegServiceI
  */
  virtual double PointBasedRegistration(const mitk::PointSet::Pointer& fixedPoints,
                                        const mitk::PointSet::Pointer& movingPoints,
                                        vtkMatrix4x4& matrix) const;

private:
  PointRegServiceRAII(const PointRegServiceRAII&); // deliberately not implemented
  PointRegServiceRAII& operator=(const PointRegServiceRAII&); // deliberately not implemented

  us::ModuleContext*                                   m_ModuleContext;
  std::vector<us::ServiceReference<PointRegServiceI> > m_Refs;
  niftk::PointRegServiceI*                             m_Service;
};

} // end namespace

#endif
