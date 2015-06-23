/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSurfaceRegServiceI_h
#define niftkSurfaceRegServiceI_h

#include <niftkIGIServicesExports.h>

#include <mitkServiceInterface.h>
#include <mitkDataNode.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
* \class SurfaceRegServiceI
* \brief Interface for a Surface Based Registration Service.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGISERVICES_EXPORT SurfaceRegServiceI
{

public:

  /**
  * \brief Does Surface Based Registration.
  * \return RMS residual error (RMS error for each movingDataSet point)
  *
  * Note: DataNode could contain mitk::PointSet, mitk::Surface or other.
  * Its up to the service to validate and throw mitk::Exception if incorrect.
  *
  * Also, implementors should consider what the implications of copying/re-formatting
  * data are. Data-sets can be large, and hence copy operations can be slow. Either
  * way, its up to the implementor of each service to consider efficiency.
  */
  virtual double SurfaceBasedRegistration(const mitk::DataNode::Pointer& fixedDataSet,
                                          const mitk::DataNode::Pointer& movingDataSet,
                                          vtkMatrix4x4& matrix) const = 0;

protected:
  SurfaceRegServiceI();
  virtual ~SurfaceRegServiceI();

private:
  SurfaceRegServiceI(const SurfaceRegServiceI&); // deliberately not implemented
  SurfaceRegServiceI& operator=(const SurfaceRegServiceI&); // deliberately not implemented
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::SurfaceRegServiceI, "uk.ac.ucl.cmic.SurfaceRegServiceI");

#endif
