/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLookupTableProviderServiceImpl_p_h
#define niftkLookupTableProviderServiceImpl_p_h

#include "niftkLookupTableProviderService.h"

#include <memory>

#include <niftkNamedLookupTableProperty.h>
#include <niftkLabeledLookupTableProperty.h>


namespace niftk
{

class LookupTableManager;

/**
 * \class LookupTableProviderServiceImpl
 * \brief Service implementation of LookupTableProviderService.
 */
class LookupTableProviderServiceImpl : public LookupTableProviderService
{
  typedef LookupTableProviderServiceImpl Self;

public:

  LookupTableProviderServiceImpl();
  virtual ~LookupTableProviderServiceImpl();

  /**
   * \see LookupTableProviderService::GetNumberOfLookupTables()
   */
  virtual unsigned int GetNumberOfLookupTables() override;

  /**
   * \see LookupTableProviderService::CheckName()
   */
  virtual bool CheckName(const QString& name) override;

  /**
   * \see LookupTableProviderService::CreateLookupTable()
   */
  virtual vtkLookupTable* CreateLookupTable(const QString& lookupTableName,
                                            float lowestValueOpacity,
                                            float highestValueOpacity) override;

  /**
   * \see LookupTableProviderService::CreateLookupTableProperty()
   */
  virtual NamedLookupTableProperty::Pointer CreateLookupTableProperty(const QString& lookupTableName,
                                                                            float lowestValueOpacity,
                                                                            float highestValueOpacity) override;

  /**
   * \see LookupTableProviderService::CreateLookupTableProperty()
   */
  virtual LabeledLookupTableProperty::Pointer CreateLookupTableProperty(const QString& lookupTableName) override;

  /**
   * \see LookupTableProviderService::AddNewLookupTableContainer()
   */
  virtual void AddNewLookupTableContainer(const LookupTableContainer* container) override;
  
  /**
   * \see LookupTableProviderService::ReplaceLookupTableContainer()
   */
  virtual void ReplaceLookupTableContainer(const LookupTableContainer* container,
                                           const QString& lookupTableName) override;

  /**
   * \see LookupTableProviderService::GetIsScaled
   */
  virtual bool GetIsScaled(const QString& lookupTableName) override;

  /**
   * \see LookupTableProviderService::GetTableNames
   */
  virtual std::vector<QString> GetTableNames() override;


  /**
   * \see Returns labels for the given table, if they exist.
   */
  LabeledLookupTableProperty::LabelListType GetLabels(const QString& lookupTableName);

  /// \brief Attempts to load LookupTable from given file, returning display name of LookupTable if successful.
  virtual QString LoadLookupTable(const QString& fileName) override;

private:

  LookupTableManager* GetManager();
  std::auto_ptr<LookupTableManager> m_Manager;

};

}

#endif
