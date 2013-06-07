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
    VTK_ICP, //VTK's ICP algorithm, point to surface
    DEFORM //A hypothetical non rigid point to surface algorithm
  };
  /**
   * \brief Write My Documentation
   */
  void Update(const mitk::Surface::Pointer fixedNode,
           const mitk::Surface::Pointer movingNode,
           vtkMatrix4x4* transformMovingToFixed);
  void Update(const mitk::PointSet::Pointer fixedNode,
           const mitk::Surface::Pointer movingNode,
           vtkMatrix4x4* transformMovingToFixed);


  /**
  * \brief Saves the given transformation to file.
  * \param[In] fileName the full absolute path of the file to be saved to, which if it already exists will be silently over-written.
  * \param[In] transform transformation matrix.
  * \return bool true if successful and false otherwise.
  */
  bool SaveToFile(const std::string& fileName, const vtkMatrix4x4& transform) const;

  /**
  * \brief Applies the given transformation to the given node.
  * \param[In] node a data node, and as each node has a mitk::Geometry3D in the mitk::BaseData, we can transfor anything.
  * \param[In] transform the VTK transformation
  * \param[In] makeUndoAble if true, use the Global Undo/Redo framework, and otherwise don't.
  * \return bool true if successful and false otherwise.
  */
      bool ApplyToNode(const mitk::DataNode::Pointer& node, vtkMatrix4x4& transform, const bool& makeUndoAble) const;


  itkSetMacro (MaximumIterations, int);
  itkSetMacro (MaximumNumberOfLandmarkPointsToUse, int);
  itkSetMacro (Method, Method);

protected:

  SurfaceBasedRegistration(); // Purposefully hidden.
  virtual ~SurfaceBasedRegistration(); // Purposefully hidden.

  SurfaceBasedRegistration(const SurfaceBasedRegistration&); // Purposefully not implemented.
  SurfaceBasedRegistration& operator=(const SurfaceBasedRegistration&); // Purposefully not implemented.

private:

  int m_MaximumIterations;
  int m_MaximumNumberOfLandmarkPointsToUse;
  Method m_Method;

  void PointSetToPolyData ( const mitk::PointSet::Pointer PointsIn, vtkPolyData* PolyOut);

  void RunVTKICP(vtkPolyData* fixedPoly,
           vtkPolyData* movingPoly,
           vtkMatrix4x4* transformMovingToFixed);
}; // end class

} // end namespace

#endif
