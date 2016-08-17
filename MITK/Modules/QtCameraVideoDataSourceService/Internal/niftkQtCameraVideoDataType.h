/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQtCameraVideoDataType_h
#define niftkQtCameraVideoDataType_h

#include <niftkIGIDataType.h>

namespace niftk
{

/**
 * \class QtCameraVideoDataType
 * \brief Class to represent video frame data from QtCamera (QCamera).
 */
class QtCameraVideoDataType : public IGIDataType
{
public:

  mitkClassMacro(QtCameraVideoDataType, IGIDataType)
  itkNewMacro(QtCameraVideoDataType)

protected:

  QtCameraVideoDataType(); // Purposefully hidden.
  virtual ~QtCameraVideoDataType(); // Purposefully hidden.

  QtCameraVideoDataType(const QtCameraVideoDataType&); // Purposefully not implemented.
  QtCameraVideoDataType& operator=(const QtCameraVideoDataType&); // Purposefully not implemented.

private:

};

} // end namespace

#endif
