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
static const int DEFAULT_TLS_ITERATIONS = 0; // Zero means 'off'.
static const int DEFAULT_TLS_PERCENTAGE = 50; // Should be (0-100].
}

/**
* \class ICPBasedRegistration
* \brief Class to perform a surface based registration of two MITK Surfaces/PointSets, using VTKs ICP.
*/
class NIFTKICPREG_EXPORT ICPBasedRegistration : public itk::Object
{
public:

  mitkClassMacroItkParent(ICPBasedRegistration, itk::Object)
  itkNewMacro(ICPBasedRegistration)

  itkSetMacro(MaximumIterations, int);
  itkSetMacro(MaximumNumberOfLandmarkPointsToUse, int);
  itkSetMacro(TLSIterations, unsigned int);
  itkSetMacro(TLSPercentage, unsigned int);

  /**
  * \brief Runs ICP registration.
  * \param fixedNode pointer to mitk::DataNode containing either mitk::Surface or mitk::Pointset.
  * \param movingNode pointer to mitk::DataNode containing either mitk::Surface or mitk::Pointset.
  * \return RMS residual for points in moving node
  */
  double Update(const mitk::DataNode::Pointer& fixedNode,
                const mitk::DataNode::Pointer& movingNode,
                vtkMatrix4x4& transformMovingToFixed,
                const mitk::DataNode::Pointer& cameraNode = mitk::DataNode::Pointer(),
                bool flipNormals = false
               );

  /**
  * \brief Runs ICP registration on vectors of fixed and moving nodes.
  * \param fixedNodes vector of pointers to mitk::DataNodes containing either mitk::Surface or mitk::Pointset.
  * \param movingNodes vector of pointers to mitk::DataNode containing either mitk::Surface or mitk::Pointset.
  * \return RMS residual for points in moving node
  */
  double Update(const std::vector<mitk::DataNode::Pointer>& fixedNodes,
                const std::vector<mitk::DataNode::Pointer>& movingNodes,
                vtkMatrix4x4& transformMovingToFixed,
                const mitk::DataNode::Pointer& cameraNode = mitk::DataNode::Pointer(),
                bool flipNormals = false
               );
protected:

  ICPBasedRegistration(); // Purposefully hidden.
  virtual ~ICPBasedRegistration(); // Purposefully hidden.

  ICPBasedRegistration(const ICPBasedRegistration&); // Purposefully not implemented.
  ICPBasedRegistration& operator=(const ICPBasedRegistration&); // Purposefully not implemented.

private:

  int          m_MaximumIterations;
  int          m_MaximumNumberOfLandmarkPointsToUse;
  unsigned int m_TLSIterations;
  unsigned int m_TLSPercentage;

  double RunVTKICP(vtkPolyData* fixedPoly,
                   vtkPolyData* movingPoly,
                   vtkMatrix4x4& transformMovingToFixed);

}; // end class

} // end namespace

#endif
