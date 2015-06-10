/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkICPBasedRegistration_h
#define niftkICPBasedRegistration_h

#include <niftkICPRegExports.h>
#include <vtkMatrix4x4.h>
#include <mitkDataNode.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace niftk {

namespace ICPBasedRegistrationConstants
{
static const int DEFAULT_MAX_ITERATIONS = 2000;
static const int DEFAULT_MAX_POINTS = 8000;
}

/**
* \class ICPBasedRegistration
* \brief Class to perform a surface based registration of two MITK Surfaces/PointSets, using VTKs ICP.
*/
class NIFTKICPREG_EXPORT ICPBasedRegistration : public itk::Object
{
public:

  mitkClassMacro(ICPBasedRegistration, itk::Object);
  itkNewMacro(ICPBasedRegistration);

  itkSetMacro(MaximumIterations, int);
  itkSetMacro(MaximumNumberOfLandmarkPointsToUse, int);
  itkSetMacro(CameraNode, mitk::DataNode::Pointer);
  itkSetMacro(FlipNormals, bool);


  /**
   * \brief Runs ICP registration.
   * \param fixedNode pointer to mitk::DataNode containing either mitk::Surface or mitk::Pointset.
   * \param movingNode pointer to mitk::DataNode containing either mitk::Surface or mitk::Pointset.
   */
  double Update(const mitk::DataNode::Pointer fixedNode,
           const mitk::DataNode::Pointer movingNode,
           vtkMatrix4x4& transformMovingToFixed);

  /**
   * \brief Generates a poly data from a mitk::DataNode.
   */
  static void NodeToPolyData (const mitk::DataNode::Pointer& node,
                              vtkPolyData& polyOut,
                              const mitk::DataNode::Pointer& cameranode = mitk::DataNode::Pointer(),
                              bool flipnormals = false);

  /**
   * \brief Generates a poly data from a mitk::PointSet.
   */
  static void PointSetToPolyData (const mitk::PointSet::Pointer& pointsIn, vtkPolyData& polyOut);

protected:

  ICPBasedRegistration(); // Purposefully hidden.
  virtual ~ICPBasedRegistration(); // Purposefully hidden.

  ICPBasedRegistration(const ICPBasedRegistration&); // Purposefully not implemented.
  ICPBasedRegistration& operator=(const ICPBasedRegistration&); // Purposefully not implemented.

private:

  int                           m_MaximumIterations;
  int                           m_MaximumNumberOfLandmarkPointsToUse;
  mitk::DataNode::Pointer       m_CameraNode;
  bool                          m_FlipNormals;

  double RunVTKICP(vtkPolyData* fixedPoly,
                   vtkPolyData* movingPoly,
                   vtkMatrix4x4& transformMovingToFixed);

}; // end class

} // end namespace

#endif
