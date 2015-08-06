/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkLabelMapReader_h
#define __mitkLabelMapReader_h


#include <mitkAbstractFileReader.h>
#include "niftkCoreGuiExports.h"
#include "QmitkLookupTableContainer.h"
#include <qfile.h>
#include <qcolor.h>

class QmitkLookupTableContainer;

namespace mitk 
{

/**
 * \class mitkLabelMapReader
 * \brief Reader for label map files. 
 * \ingroup IO
 */
class NIFTKCOREGUI_EXPORT LabelMapReader : public AbstractFileReader
{

public: 

  LabelMapReader();
  LabelMapReader(const LabelMapReader & other);
  virtual LabelMapReader * Clone() const;
  virtual ~LabelMapReader(){};
  
  using mitk::AbstractFileReader::Read;
  
  /**
   * \brief Read the file and return mitkBaseData 
   */
  virtual std::vector<itk::SmartPointer<BaseData> > Read();  
    
  // get a QmitkLookupTableContainer from a file
  virtual QmitkLookupTableContainer* GetLookupTableContainer();
 
  /**
   * \brief Set the order to assign the QmitkLookupTableContainer as 
   *  the label map file does not store this information 
   */
  void SetOrder(int order){m_Order = order;}

  void SetQFile(QFile &file){m_InputQFile = &file;};

private: 

  bool ReadLabelMap(std::istream &);  
  
  QmitkLookupTableContainer::LabelListType m_Labels;

  typedef std::vector<QColor> ColorListType;
  ColorListType m_Colors;

  us::ServiceRegistration<mitk::IFileReader> m_ServiceReg;

  int m_Order;
  QString m_DisplayName;
  QFile* m_InputQFile;
};

} // namespace mitk

#endif // __mitkLabelMapReader_h