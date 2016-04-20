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
#include <niftkNamedLookupTableProperty.h>
#include <niftkLabeledLookupTableProperty.h>
#include <memory>

class QmitkLookupTableManager;

/**
 * \class QmitkLookupTableProviderServiceImpl
 * \brief Service implementation of QmitkLookupTableProviderService.
 */
class QmitkLookupTableProviderServiceImpl : public QmitkLookupTableProviderService 
{

public:

  QmitkLookupTableProviderServiceImpl();
  virtual ~QmitkLookupTableProviderServiceImpl();

  /**
   * \see QmitkLookupTableProviderService::GetNumberOfLookupTables()
   */
  virtual unsigned int GetNumberOfLookupTables() override;

  /**
   * \see QmitkLookupTableProviderService::CheckName()
   */
  virtual bool CheckName(const QString& name) override;

  /**
   * \see QmitkLookupTableProviderService::CreateLookupTable()
   */
  virtual vtkLookupTable* CreateLookupTable(const QString& lookupTableName,
                                            float lowestValueOpacity,
                                            float highestValueOpacity) override;

  /**
   * \see QmitkLookupTableProviderService::CreateLookupTableProperty()
   */
  virtual niftk::NamedLookupTableProperty::Pointer CreateLookupTableProperty(const QString& lookupTableName,
                                                                            float lowestValueOpacity,
                                                                            float highestValueOpacity) override;
  
  /**
   * \see QmitkLookupTableProviderService::CreateLookupTableProperty()
   */
  virtual niftk::LabeledLookupTableProperty::Pointer CreateLookupTableProperty(const QString& lookupTableName) override;

  /**
   * \see QmitkLookupTableProviderService::AddNewLookupTableContainer()
   */
  virtual void AddNewLookupTableContainer(const QmitkLookupTableContainer* container) override;
  
  /**
   * \see QmitkLookupTableProviderService::ReplaceLookupTableContainer()
   */
  virtual void ReplaceLookupTableContainer(const QmitkLookupTableContainer* container, const QString& lookupTableName) override;

  /**
   * \see QmitkLookupTableProviderService::GetIsScaled
   */
  virtual bool GetIsScaled(const QString& lookupTableName) override;

  /**
   * \see QmitkLookupTableProviderService::GetTableNames
   */
  virtual std::vector<QString> GetTableNames() override;


  /**
   * \see Returns labels for the given table, if they exist.
   */
  niftk::LabeledLookupTableProperty::LabelListType GetLabels(const QString& lookupTableName);

private:

  QmitkLookupTableManager* GetManager();
  std::auto_ptr<QmitkLookupTableManager> m_Manager;
  
};

#endif
