/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUltrasonixDataType_h
#define niftkUltrasonixDataType_h

#include <niftkIGIDataType.h>
#include <cv.h>

namespace niftk
{

/**
 * \class UltrasonixDataType
 * \brief Class to represent ultrasound frame data from Ultrasonix.
 */
class UltrasonixDataType : public IGIDataType
{
public:

  mitkClassMacro(UltrasonixDataType, IGIDataType)
  itkNewMacro(UltrasonixDataType)

  /**
  * \brief Used for loading in an image.
  */
  void CloneImage(const IplImage *image);

  /**
  * \brief Returns the internal image, so do not modify it.
  */
  const IplImage* GetImage();

protected:

  UltrasonixDataType(); // Purposefully hidden.
  virtual ~UltrasonixDataType(); // Purposefully hidden.

  UltrasonixDataType(const UltrasonixDataType&); // Purposefully not implemented.
  UltrasonixDataType& operator=(const UltrasonixDataType&); // Purposefully not implemented.

private:

  IplImage *m_Image;

};

} // end namespace

#endif
