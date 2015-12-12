/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQtAudioDataType_h
#define niftkQtAudioDataType_h

#include <niftkIGIDataType.h>

namespace niftk
{

/**
 * \class QtAudioDataType
 * \brief Class to represent ultrasound frame data from QtAudio.
 */
class QtAudioDataType : public IGIDataType
{
public:

  mitkClassMacro(QtAudioDataType, IGIDataType);
  itkNewMacro(QtAudioDataType);

protected:

  QtAudioDataType(); // Purposefully hidden.
  virtual ~QtAudioDataType(); // Purposefully hidden.

  QtAudioDataType(const QtAudioDataType&); // Purposefully not implemented.
  QtAudioDataType& operator=(const QtAudioDataType&); // Purposefully not implemented.

private:

};

} // end namespace

#endif
