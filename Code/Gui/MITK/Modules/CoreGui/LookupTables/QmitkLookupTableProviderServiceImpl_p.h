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
   * \see QmitkLookupTableProviderService::CreateLookupTable()
   */
  virtual vtkLookupTable* CreateLookupTable(unsigned int lookupTableIndex,
                                            float lowestValueOpacity,
                                            float highestValueOpacity);

  /**
   * \see QmitkLookupTableProviderService::CreateLookupTableProperty()
   */
  virtual mitk::NamedLookupTableProperty::Pointer CreateLookupTableProperty(unsigned int lookupTableIndex,
                                                                            float lowestValueOpacity,
                                                                            float highestValueOpacity);

  /**
   * \see QmitkLookupTableProviderService::GetName
   */
  std::string GetName(unsigned int lookupTableIndex);


  /**
   * \see QmitkLookupTableProviderService::GetIsScaled
   */
  bool GetIsScaled(unsigned int lookupTableIndex);

private:
  QmitkLookupTableManager* GetManager();
  std::auto_ptr<QmitkLookupTableManager> m_Manager;
};

#endif
