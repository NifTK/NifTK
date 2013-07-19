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
   * \brief Main registration method.
   * \return the fiducial registration error
   */
  double Update(
      mitk::DataStorage::Pointer& dataStorage,
      mitk::PointSet::Pointer& tagPointSet,
      mitk::PointSet::Pointer& tagNormals,
      mitk::DataNode::Pointer& modelNode,
      const std::string& transformNodeToUpdate,
      const bool useNormals,
      vtkMatrix4x4& outputTransform) const;

protected:

  TagTrackingRegistrationManager(); // Purposefully hidden.
  virtual ~TagTrackingRegistrationManager(); // Purposefully hidden.

  TagTrackingRegistrationManager(const TagTrackingRegistrationManager&); // Purposefully not implemented.
  TagTrackingRegistrationManager& operator=(const TagTrackingRegistrationManager&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
