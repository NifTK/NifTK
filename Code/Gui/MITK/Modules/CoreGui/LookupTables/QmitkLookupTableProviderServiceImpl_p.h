/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkLookupTableProviderServiceImpl_p_h
#define QmitkLookupTableProviderServiceImpl_p_h

#include "QmitkLookupTableProviderService.h"
#include <mitkNamedLookupTableProperty.h>
#include <mitkLabeledLookupTableProperty.h>
#include <memory>

class QmitkLookupTableManager;

/**
 * \class QmitkLookupTableProviderServiceImpl
 * \brief Service implementation of QmitkLookupTableProviderService.
 */
class QmitkLookupTableProviderServiceImpl : public QmitkLookupTableProviderService {

public:

  QmitkLookupTableProviderServiceImpl();
  virtual ~QmitkLookupTableProviderServiceImpl();

  /**
   * \see QmitkLookupTableProviderService::GetNumberOfLookupTables()
   */
  virtual unsigned int GetNumberOfLookupTables();

  /**
   * \see QmitkLookupTableProviderService::CheckName()
   */
  virtual bool CheckName(QString& name);

  /**
   * \see QmitkLookupTableProviderService::CreateLookupTable()
   */
  virtual vtkLookupTable* CreateLookupTable(QString& lookupTableName,
                                            float lowestValueOpacity,
                                            float highestValueOpacity);

  /**
   * \see QmitkLookupTableProviderService::CreateLookupTableProperty()
   */
  virtual mitk::NamedLookupTableProperty::Pointer CreateLookupTableProperty(QString& lookupTableName,
                                                                            float lowestValueOpacity,
                                                                            float highestValueOpacity);
  
  /**
   * \see QmitkLookupTableProviderService::CreateLookupTableProperty()
   */
  virtual mitk::LabeledLookupTableProperty::Pointer CreateLookupTableProperty(QString& lookupTableName);

  /**
   * \see QmitkLookupTableProviderService::AddNewLookupTableContainer()
   */
  virtual void AddNewLookupTableContainer(QmitkLookupTableContainer* container);
  
  /**
   * \see QmitkLookupTableProviderService::ReplaceLookupTableContainer()
   */
  virtual void ReplaceLookupTableContainer(QmitkLookupTableContainer* container, QString& lookupTableName);

  /**
   * \see QmitkLookupTableProviderService::GetIsScaled
   */
  virtual bool GetIsScaled(QString& lookupTableName);

  /**
   * \see QmitkLookupTableProviderService::GetTableNames
   */
  virtual std::vector<QString> GetTableNames();


  /**
   * \see Returns labels for the given table, if they exist.
   */
  mitk::LabeledLookupTableProperty::LabelListType GetLabels(QString& lookupTableName);

private:
  QmitkLookupTableManager* GetManager();
  std::auto_ptr<QmitkLookupTableManager> m_Manager;
};

#endif
