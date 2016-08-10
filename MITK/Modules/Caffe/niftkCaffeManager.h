/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCaffeManager_h
#define niftkCaffeManager_h

#include "niftkCaffeExports.h"
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkCommon.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>

namespace niftk
{

/**
 * \class CaffeManager
 * \brief Manager class to coordinate Caffe segmentation.
 *
 * All errors must be thrown as a subclass of mitk::Exception.
 *
 * The class is stateful. It is assumed that the constructor will
 * initialise the network, so if the constructor is successful,
 * the class should be ready to segment. Then all subsequent
 * calls to the Segment method will compute a segmented image.
 */
class NIFTKCAFFE_EXPORT CaffeManager : public itk::Object
{
public:

  mitkClassMacroItkParent(CaffeManager, itk::Object)
  mitkNewMacro2Param(CaffeManager, const std::string&, const std::string&)

  void Segment(const mitk::DataStorage::ConstPointer& dataStorage,
               const mitk::DataNode::ConstPointer& inputImage);

protected:

  CaffeManager(const std::string& networkDescriptionFileName,  // Purposefully hidden.
               const std::string& networkWeightsFileName
               );
  virtual ~CaffeManager();                                     // Purposefully hidden.

  CaffeManager(const CaffeManager&);                           // Purposefully not implemented.
  CaffeManager& operator=(const CaffeManager&);                // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
