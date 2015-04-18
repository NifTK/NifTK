/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPointRegServiceI_h
#define niftkPointRegServiceI_h

#include <niftkIGIExports.h>

#include <mitkServiceInterface.h>
#include <mitkPointSet.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
* \class Interface for a Point Based Registration Service.
*/
class NIFTKIGI_EXPORT PointRegServiceI
{

public:

  /**
  * @brief Does Point Based Registration of equal length, corresponding, ordered point sets.
  */
  virtual double PointBasedRegistration(const mitk::PointSet::Pointer& fixedPoints,
                                        const mitk::PointSet::Pointer& movingPoints,
                                        vtkMatrix4x4& matrix) const = 0;

protected:
  PointRegServiceI();
  virtual ~PointRegServiceI();

private:
  PointRegServiceI(const PointRegServiceI&); // deliberately not implemented
  PointRegServiceI& operator=(const PointRegServiceI&); // deliberately not implemented
};

} // end namespace

MITK_DECLARE_SERVICE_INTERFACE(niftk::PointRegServiceI, "uk.ac.ucl.cmic.PointRegServiceI");

#endif
