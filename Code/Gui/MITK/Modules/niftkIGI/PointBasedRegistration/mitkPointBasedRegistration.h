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
 * This is two different use-cases, and the usage is quite different.
 * The code is kept here in one class for convenience to the user.
 *
 * The first case is for PointRegView, where expected usage is:
 *
 * <pre>
 * mitk::PointBasedRegistration::Pointer reg = mitk::PointBasedRegistration::New();
 * reg->SetUseICPInitialisation(bool);     // user can pick true/false.
 * reg->SetUsePointIDToMatchPoints(false); // must be false.
 * reg->Update(..);
 * </pre>
 * and the code calls the underlying mitk::NavigationDataLandmarkTransformFilter.
 *
 * If <code>UseICPInitialisation</code> is false then there must be 3 or more points
 * in each point set, the fixed and moving point set must be the same size, and both
 * ordered such that corresponding points are at a given index.
 *
 * If <code>UseICPInitialisation</code> is true, then there must be at least 6 points
 * in each point set, and the algorithm searches for closest matching points to
 * do a point based registration.
 *
 * The second use case is for Tag Tracker View.
 *
 * <pre>
 * mitk::PointBasedRegistration::Pointer reg = mitk::PointBasedRegistration::New();
 * reg->SetUsePointIDToMatchPoints(true); // must be true.
 * reg->Update(..);
 * </pre>
 * in this case the ICPInitialisation is ignored.
 *
 * The algorithm takes the fixed and moving point set, and looks at the point's ID,
 * and extracts a list of only those point IDs that match, and then performs a standard
 * point based registration using mitk::NavigationDataLandmarkTransformFilter.
 * This will require 3 or more matched points in each point set.
 */
class NIFTKIGI_EXPORT PointBasedRegistration : public itk::Object
{
public:

  mitkClassMacro(PointBasedRegistration, itk::Object);
  itkNewMacro(PointBasedRegistration);

  /**
   * \brief Stores the default value of UseICPInitialisation = false.
   */
  static const bool DEFAULT_USE_ICP_INITIALISATION;

  /**
   * \brief Stores the default value of UsePointIDToMatchPoints = false.
   */
  static const bool DEFAULT_USE_POINT_ID_TO_MATCH;

  itkSetMacro(UseICPInitialisation, bool);
  itkGetMacro(UseICPInitialisation, bool);

  itkSetMacro(UsePointIDToMatchPoints, bool);
  itkGetMacro(UsePointIDToMatchPoints, bool);

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
              vtkMatrix4x4& outputTransform) const;

protected:

  PointBasedRegistration(); // Purposefully hidden.
  virtual ~PointBasedRegistration(); // Purposefully hidden.

  PointBasedRegistration(const PointBasedRegistration&); // Purposefully not implemented.
  PointBasedRegistration& operator=(const PointBasedRegistration&); // Purposefully not implemented.

private:

  bool m_UseICPInitialisation;
  bool m_UsePointIDToMatchPoints;

}; // end class

} // end namespace

#endif
