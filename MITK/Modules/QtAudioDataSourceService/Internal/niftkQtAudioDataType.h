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

  QtAudioDataType();
  virtual ~QtAudioDataType();
  QtAudioDataType(const QtAudioDataType&);             // Copy constructor
  QtAudioDataType& operator=(const QtAudioDataType&);  // Copy assignment
  QtAudioDataType(QtAudioDataType&&);                  // Move constructor
  QtAudioDataType& operator=(QtAudioDataType&&);       // Move assignment

  void SetBlob(char* blob, std::size_t length);
  std::pair<char*, std::size_t> GetBlob() const;

private:
  char*       m_AudioBlob;
  std::size_t m_Length;
};

} // end namespace

#endif
