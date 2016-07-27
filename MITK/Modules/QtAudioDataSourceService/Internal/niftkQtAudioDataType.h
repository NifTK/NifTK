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
 * \brief Class to represent audio data from QtAudio.
 */
class QtAudioDataType : public IGIDataType
{
public:

  mitkClassMacro(QtAudioDataType, IGIDataType)
  itkNewMacro(QtAudioDataType)

  void SetBlob(const char* blob, std::size_t length);
  std::pair<const char*, std::size_t> GetBlob() const;

protected:

  QtAudioDataType(); // Purposefully hidden.
  virtual ~QtAudioDataType(); // Purposefully hidden.

  QtAudioDataType(const QtAudioDataType&); // Purposefully not implemented.
  QtAudioDataType& operator=(const QtAudioDataType&); // Purposefully not implemented.

private:
  const char*   m_AudioBlob;
  std::size_t   m_Length;
};

} // end namespace

#endif
