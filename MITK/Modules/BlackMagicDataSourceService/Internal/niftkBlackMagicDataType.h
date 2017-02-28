/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBlackMagicDataType_h
#define niftkBlackMagicDataType_h

#include <niftkIGIDataType.h>

namespace niftk
{

/**
 * \class BlackMagicDataType
 * \brief Class to hold video frame data from BlackMagic.
 *
 * Data types can be whatever you like. cv::Mat, IplImage etc.
 */
class BlackMagicDataType : public IGIDataType
{
public:


  BlackMagicDataType();
  virtual ~BlackMagicDataType();
  BlackMagicDataType(const BlackMagicDataType&);             // Copy constructor
  BlackMagicDataType& operator=(const BlackMagicDataType&);  // Copy assignment
  BlackMagicDataType(BlackMagicDataType&&);                  // Move constructor
  BlackMagicDataType& operator=(BlackMagicDataType&&);       // Move assignment

private:

  // Use any data type you like.
  // Just make sure copy/move works.
  // IplImage *m_Image;

};

} // end namespace

#endif
