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
#include "mitkLabeledLookupTableProperty.h"
#include <QFile.h>
#include <QColor.h>

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
    
  /**
   * \brief Get a QmitkLookupTableContainer from file 
   */
  virtual QmitkLookupTableContainer* GetLookupTableContainer();
 
  /**
   * \brief Set the order to assign the QmitkLookupTableContainer as 
   *  the label map file does not store this information 
   */
  void SetOrder(int order){m_Order = order;}

  /**
   * \brief Set the file to read the label information from
   */
  void SetQFile(QFile &file){m_InputQFile = &file;};

private: 

  /** \brief Parse the istream to determine the labels, colors. */
  bool ReadLabelMap(std::istream &);  

  /** To temporarily store the labels (pixel value/name). */
  LabeledLookupTableProperty::LabelListType m_Labels;

  typedef std::vector<QColor> ColorListType;
  /** To temporarily store the list of colors. */
  ColorListType m_Colors;

  us::ServiceRegistration<mitk::IFileReader> m_ServiceReg;

  /** To temporarily store the order. */
  int m_Order;

  /** To temporarily store the display name. */
  QString m_DisplayName;

  /** To store the QFile to read the labels from. */
  QFile* m_InputQFile;

};

} // namespace mitk

#endif // __mitkLabelMapReaderService_h
