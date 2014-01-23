/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkSurfaceBasedRegistration_h
#define mitkSurfaceBasedRegistration_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <vtkMatrix4x4.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace mitk {

/**
 * \class SurfaceBasedRegistration
 * \brief Class to perform a surface based registration of two MITK Surfaces/PointSets.
 */
class NIFTKIGI_EXPORT SurfaceBasedRegistration : public itk::Object
{
public:

  mitkClassMacro(SurfaceBasedRegistration, itk::Object);
  itkNewMacro(SurfaceBasedRegistration);

  enum Method 
  {
    VTK_ICP, // VTK's ICP algorithm, point to surface
    DEFORM   // A hypothetical non rigid point to surface algorithm
  };

  static const int DEFAULT_MAX_ITERATIONS;
  static const int DEFAULT_MAX_POINTS;
  static const bool DEFAULT_USE_DEFORMABLE;

  itkSetMacro (MaximumIterations, int);
  itkSetMacro (MaximumNumberOfLandmarkPointsToUse, int);
  itkSetMacro (Method, Method);
  itkSetMacro(CameraNode, mitk::DataNode::Pointer);
  itkSetMacro(FlipNormals, bool);

  /**
   * \brief Write My Documentation
   */
  void Update(const mitk::DataNode::Pointer fixedNode,
           const mitk::DataNode::Pointer movingNode,
           vtkMatrix4x4& transformMovingToFixed);

  /**
   * \brief Generates a poly data from a mitk::DataNode.
   */
  static void NodeToPolyData ( const mitk::DataNode::Pointer& node , vtkPolyData& polyOut, const mitk::Geometry3D::Pointer& cameranode = mitk::Geometry3D::Pointer());

  /**
   * \brief Generates a poly data from a mitk::PointSet.
   */
  static void PointSetToPolyData ( const mitk::PointSet::Pointer& pointsIn, vtkPolyData& polyOut);

protected:

  SurfaceBasedRegistration(); // Purposefully hidden.
  virtual ~SurfaceBasedRegistration(); // Purposefully hidden.

  SurfaceBasedRegistration(const SurfaceBasedRegistration&); // Purposefully not implemented.
  SurfaceBasedRegistration& operator=(const SurfaceBasedRegistration&); // Purposefully not implemented.

private:

  int m_MaximumIterations;
  int m_MaximumNumberOfLandmarkPointsToUse;
  Method m_Method;

  mitk::DataNode::Pointer     m_CameraNode;
  bool                        m_FlipNormals;

  vtkSmartPointer<vtkMatrix4x4> m_Matrix;

  void RunVTKICP(vtkPolyData* fixedPoly,
           vtkPolyData* movingPoly,
           vtkMatrix4x4& transformMovingToFixed);

}; // end class

} // end namespace

#endif
