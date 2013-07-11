/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkPointBasedRegistration_h
#define mitkPointBasedRegistration_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <vtkMatrix4x4.h>
#include <mitkPointSet.h>
#include <mitkDataNode.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class PointBasedRegistration
 * \brief Class to implement point based registration of two point sets.
 *
 * This class is called from both PointRegView and TagTrackerView.
 * There are two 'normal' modes of operation:
 * <ol>
 * <li>Exact corresponding points: Both point sets should be ordered, the same size, and corresponding point-wise. Needs at least 3 points</li>
 * <li>ICP mode: Calculates closest points to initialise, and needs at least 6 points (for some reason in MITK).</li>
 * </ol>
 * Due to the use of mitk::PointSet where point numbers can be labelled, we can also have a fallback position for point based registration,
 * and extract points with matching ID. In this case, if we get 3 or more points with matching ID, we can do
 * a straight point based match, with corresponding points. By default, this is a fallback for when the first
 * two options cannot be performed. By setting AlwaysTryMatchedPoints to true, the points will always
 * be filtered, and if the result has lower FRE will be used in preference.
 */
class NIFTKIGI_EXPORT PointBasedRegistration : public itk::Object
{
public:

  mitkClassMacro(PointBasedRegistration, itk::Object);
  itkNewMacro(PointBasedRegistration);

  /**
   * \brief Stores the default value of whether to use ICP initialisation = false.
   */
  static const bool DEFAULT_USE_ICP_INITIALISATION;

  itkSetMacro(AlwaysTryMatchedPoints, bool);
  itkGetMacro(AlwaysTryMatchedPoints, bool);

  /**
   * \brief Main method to calculate the point based registration.
   * \param[In] fixedPointSet a point set
   * \param[In] movingPointSet a point set
   * \param[In,Out] useICPInitialisation if true, will compute closest point pairs, so the
   * number of points in each data set can be different, but does require at least 6 points in
   * each data set, and if false will assume that the point sets are ordered, of equal size,
   * and with points corresponding.
   * \param[In,Out] the transformation to transform the moving point set into the coordinate system of the fixed point set.
   * \return Returns the Fiducial Registration Error
   */
  double Update(const mitk::PointSet::Pointer fixedPointSet,
              const mitk::PointSet::Pointer movingPointSet,
              const bool& useICPInitialisation,
              vtkMatrix4x4& outputTransform) const;

protected:

  PointBasedRegistration(); // Purposefully hidden.
  virtual ~PointBasedRegistration(); // Purposefully hidden.

  PointBasedRegistration(const PointBasedRegistration&); // Purposefully not implemented.
  PointBasedRegistration& operator=(const PointBasedRegistration&); // Purposefully not implemented.

private:

  bool m_AlwaysTryMatchedPoints;

}; // end class

} // end namespace

#endif
