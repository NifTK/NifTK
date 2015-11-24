/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIFilePerFrameDataType_h
#define niftkIGIFilePerFrameDataType_h

#include "niftkIGIDataSourcesExports.h"
#include "niftkIGIDataType.h"

namespace niftk
{

/**
* \class IGIFilePerFrameDataType
* \brief Abstract base class for IGI Data, such as objects
* containing tracking data, video frames or ultrasound frames.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIFilePerFrameDataType : public niftk::IGIDataType
{
public:

  mitkClassMacroItkParent(IGIFilePerFrameDataType, niftk::IGIDataType);
  itkNewMacro(IGIFilePerFrameDataType);

  itkSetMacro(IsSaved, bool);
  itkGetMacro(IsSaved, bool);

  itkSetMacro(ShouldBeSaved, bool);
  itkGetMacro(ShouldBeSaved, bool);

  itkSetMacro(FileName, std::string);
  itkGetMacro(FileName, std::string);

protected:

  IGIFilePerFrameDataType(); // Purposefully hidden.
  virtual ~IGIFilePerFrameDataType(); // Purposefully hidden.

  IGIFilePerFrameDataType(const IGIFilePerFrameDataType&); // Purposefully not implemented.
  IGIFilePerFrameDataType& operator=(const IGIFilePerFrameDataType&); // Purposefully not implemented.

private:

  bool         m_IsSaved;
  bool         m_ShouldBeSaved;
  std::string  m_FileName;
};

} // end namespace

#endif
