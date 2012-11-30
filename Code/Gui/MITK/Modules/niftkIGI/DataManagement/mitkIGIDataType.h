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
 */
class NIFTKIGI_EXPORT IGIDataType : public itk::Object
{
public:

  mitkClassMacro(IGIDataType, itk::Object);
  itkNewMacro(IGIDataType);

  itkSetMacro(DataSource, std::string);
  itkGetMacro(DataSource, std::string);

  igtlUint64 GetTimeStampUint64() const { return this->m_TimeStamp->GetTimeStampUint64(); }
  void SetTimeStampUint64(const igtlUint64& time) { this->m_TimeStamp->SetTime(time); this->Modified(); }

  itkSetMacro(Duration, igtlUint64);
  itkGetMacro(Duration, igtlUint64);

  itkSetMacro(FrameId, unsigned long int);
  itkGetMacro(FrameId, unsigned long int);

  itkSetMacro(IsSaved, bool);
  itkGetMacro(IsSaved, bool);

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
  igtlUint64 m_Duration;
  unsigned long int m_FrameId;
  bool m_IsSaved;
};

} // end namespace

#endif // MITKIGIDATATYPE_H
