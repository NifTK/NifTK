/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKIGIDATATYPE_H
#define MITKIGIDATATYPE_H

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <igtlTimeStamp.h>

namespace mitk
{

/**
 * \class IGIDataType
 * \brief Abstract base class for IGI Data, such as messages containing tracking data or video frames.
 *
 * NOTE: All timestamps should be in UTC format. Also, take care NOT to expose a pointer to the
 * igtl::TimeStamp object. You should only ever expose a copy of this data, or an equivalent
 * representation of it, i.e. if you Set/Get the igtlUint64 values, then NO-ONE can modify the timestamp
 * and set the time to TAI for example.
 */
class NIFTKIGI_EXPORT IGIDataType : public itk::Object
{
public:

  mitkClassMacro(IGIDataType, itk::Object);
  itkNewMacro(IGIDataType);

  itkSetMacro(DataSource, std::string);
  itkGetMacro(DataSource, std::string);

  igtlUint64 GetTimeStampInNanoSeconds() const;
  void SetTimeStampInNanoSeconds(const igtlUint64& time);

  itkSetMacro(Duration, unsigned long int);
  itkGetMacro(Duration, unsigned long int);

  itkSetMacro(FrameId, unsigned long int);
  itkGetMacro(FrameId, unsigned long int);

  itkSetMacro(IsSaved, bool);
  itkGetMacro(IsSaved, bool);

  itkSetMacro(ShouldBeSaved, bool);
  itkGetMacro(ShouldBeSaved, bool);

  itkSetMacro(FileName, std::string);
  itkGetMacro(FileName, std::string);

  /**
   * \brief This object can contain any data, and derived classes should override this.
   */
  virtual void* GetData() const { return NULL; }

protected:

  IGIDataType(); // Purposefully hidden.
  virtual ~IGIDataType(); // Purposefully hidden.

  IGIDataType(const IGIDataType&); // Purposefully not implemented.
  IGIDataType& operator=(const IGIDataType&); // Purposefully not implemented.

private:

  std::string m_DataSource;
  igtl::TimeStamp::Pointer m_TimeStamp;
  unsigned long int m_Duration;
  unsigned long int m_FrameId;
  bool m_IsSaved;
  bool m_ShouldBeSaved;
  std::string m_FileName;
};

} // end namespace

#endif // MITKIGIDATATYPE_H
