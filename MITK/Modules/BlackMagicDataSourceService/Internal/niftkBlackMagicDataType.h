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

  mitkClassMacro(BlackMagicDataType, IGIDataType)
  itkNewMacro(BlackMagicDataType)

  /**
  * \brief Used for loading in an image.
  */
  //void CloneImage(const IplImage *image);

  /**
  * \brief Returns the internal image, so do not modify it.
  */
  //const IplImage* GetImage();

protected:

  BlackMagicDataType(); // Purposefully hidden.
  virtual ~BlackMagicDataType(); // Purposefully hidden.

  BlackMagicDataType(const BlackMagicDataType&); // Purposefully not implemented.
  BlackMagicDataType& operator=(const BlackMagicDataType&); // Purposefully not implemented.

private:

  // Use any data type you like.
  // IplImage *m_Image;

};

} // end namespace

#endif
