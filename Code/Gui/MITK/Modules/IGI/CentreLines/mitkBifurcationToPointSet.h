/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkBifurcationToPointSet_h
#define mitkBifurcationToPointSet_h

#include "niftkIGIExports.h"
#include <vtkPolyData.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class BifurcationToPointSet
 * \brief Takes a vector of vtkPolyData pointers and computes an mitk::PointSet representing the bifurcations.
 *
 * The vtkPolyData are assumed to be such as those produced by vmtkcentrelines.
 *   - Containing VTK LINE
 *   - Centre lines start at a single seed point, and follow all the way to a target point.
 *   - i.e. the DONT branch off of each other.
 */
class NIFTKIGI_EXPORT BifurcationToPointSet : public itk::Object
{
public:

  mitkClassMacro(BifurcationToPointSet, itk::Object);
  itkNewMacro(BifurcationToPointSet);

  /**
   * \brief Computes the mitk::PointSet representing bifurcations.
   */
  void Update(const std::vector<vtkPolyData*> polyDatas,
              mitk::PointSet& pointSet
             );

protected:

  BifurcationToPointSet(); // Purposefully hidden.
  virtual ~BifurcationToPointSet(); // Purposefully hidden.

  BifurcationToPointSet(const BifurcationToPointSet&); // Purposefully not implemented.
  BifurcationToPointSet& operator=(const BifurcationToPointSet&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
