/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkLookupTableProviderService_h
#define QmitkLookupTableProviderService_h

#include <vector>
#include <string>
#include <mitkNamedLookupTableProperty.h>

// Microservices
#include <usServiceInterface.h>

class vtkLookupTable;

/**
 * \class QmitkLookupTableProviderService
 * \brief Service to provide lookup tables.
 */
struct QmitkLookupTableProviderService
{
  /**
   * \brief Returns the number of lookup tables that this service knows about.
   */
  virtual unsigned int GetNumberOfLookupTables() = 0;

  /**
   * \brief Returns a pointer to a new instance of a lookup table, as specified by the index, which the client is then responsible for deleting.
   * \param lookupTableIndex a positive integer in the range [0.. (this->GetNumberOfLookupTables() - 1)].
   * \param lowestValueOpacity opacity value in the range [0..1], which if outside this range, is clamped.
   * \param highestValueOpacity opacity value in the range [0..1], which if outside this range, is clamped.
   * \return
   */
  virtual vtkLookupTable* CreateLookupTable(unsigned int lookupTableIndex,
                                            float lowestValueOpacity,
                                            float highestValueOpacity) = 0;

  /**
   * \brief Same, as CreateLookupTable, but wraps it into a mitk::NamedLookupTableProperty.
   */
  virtual mitk::NamedLookupTableProperty::Pointer CreateLookupTableProperty(unsigned int lookupTableIndex,
                                                                            float lowestValueOpacity,
                                                                            float highestValueOpacity) = 0;

  /**
   * \brief Returns the display name of the given table.
   */
  std::string GetName(unsigned int lookupTableIndex);
};

US_DECLARE_SERVICE_INTERFACE(QmitkLookupTableProviderService, "QmitkLookupTableProviderService/1.0")

#endif
