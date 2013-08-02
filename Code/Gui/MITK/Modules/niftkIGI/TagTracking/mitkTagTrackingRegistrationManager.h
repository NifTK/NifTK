/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTagTrackingRegistrationManager_h
#define mitkTagTrackingRegistrationManager_h

#include "niftkIGIExports.h"
#include <vtkMatrix4x4.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class TagTrackingRegistrationManager
 * \brief Manager class to coordinate mitk::PointBasedRegistration and mitk::PointAndNormalsBasedRegistration
 * for the purpose of tag tracking.
 */
class NIFTKIGI_EXPORT TagTrackingRegistrationManager : public itk::Object
{
public:

  mitkClassMacro(TagTrackingRegistrationManager, itk::Object);
  itkNewMacro(TagTrackingRegistrationManager);

  /**
   * \brief This plugin creates its own data node to store a point set, this static variable stores the name.
   */
  static const char* POINTSET_NODE_ID;

  /**
   * \brief This plugin creates its own data node to store a point set, this static variable stores the name.
   */
  static const char* TRANSFORM_NODE_ID;


  /**
   * \brief Main update method to compute the transform.
   * \param[In] dataStorage we look up nodes in data storage
   * \param[In] tagPointSet the reconstructed tag positions
   * \param[In] tagNormals the reconstructed surface normals for those positions
   * \param[In] modelNode the model can be either a mitk::Surface which must have normals and scalars containing pointIDs, or a mitk::PointSet.
   * \param[In] transformNodeToUpdate in addition to passing back the outputTransform, this method can directly update an existing mitk::CoordinateAxesData node in Data Storage, this is the name of it.
   * \param[Out] outputTransform output transformation
   * \param[Out] fiducialRegistrationError the Fiducial Registration Error
   */
  bool Update(
      mitk::DataStorage::Pointer& dataStorage,
      mitk::PointSet::Pointer& tagPointSet,
      mitk::PointSet::Pointer& tagNormals,
      mitk::DataNode::Pointer& modelNode,
      const std::string& transformNodeToUpdate,
      const bool useNormals,
      vtkMatrix4x4& outputTransform,
      double& fiducialRegistrationError) const;

protected:

  TagTrackingRegistrationManager(); // Purposefully hidden.
  virtual ~TagTrackingRegistrationManager(); // Purposefully hidden.

  TagTrackingRegistrationManager(const TagTrackingRegistrationManager&); // Purposefully not implemented.
  TagTrackingRegistrationManager& operator=(const TagTrackingRegistrationManager&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
