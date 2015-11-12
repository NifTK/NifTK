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
#include <mitkLabeledLookupTableProperty.h>

// Microservices
#include <mitkServiceInterface.h>

class vtkLookupTable;
class QmitkLookupTableContainer;

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
   * \brief Returns if a LookupTable with the given name exists.
   */
  virtual bool CheckName(const QString& name) = 0;

  /**
   * \brief Returns a pointer to a new instance of a lookup table, as specified by the index, which the client is then responsible for deleting.
   * \param lookupTableIndex a positive integer in the range [0.. (this->GetNumberOfLookupTables() - 1)].
   * \param lowestValueOpacity opacity value in the range [0..1], which if outside this range, is clamped.
   * \param highestValueOpacity opacity value in the range [0..1], which if outside this range, is clamped.
   * \return
   */
  virtual vtkLookupTable* CreateLookupTable(const QString& lookupTableName,
                                            float lowestValueOpacity,
                                            float highestValueOpacity) = 0;

  /**
   * \brief Same as CreateLookupTable, but wraps it into a mitk::NamedLookupTableProperty.
   */
  virtual mitk::NamedLookupTableProperty::Pointer CreateLookupTableProperty(const QString& lookupTableName,
                                                                            float lowestValueOpacity,
                                                                            float highestValueOpacity) = 0;

  /**
   * \brief Similar to CreateLookupTable, but create a mitk::LabeledLookupTableProperty.
   */
  virtual mitk::LabeledLookupTableProperty::Pointer CreateLookupTableProperty(const QString& lookupTableName) = 0;

  /**
   * \brief Add a new LookupTableContainer to the LookupTableManager.
   */
  virtual void AddNewLookupTableContainer(QmitkLookupTableContainer* container) = 0;

  /**
   * \brief Replace the LookupTableContainer of the given name with another.
   */
  virtual void ReplaceLookupTableContainer(QmitkLookupTableContainer* container, const QString& lookupTableName) = 0;

  /**
   * \brief Returns the display names of  all table.
   */
  virtual std::vector<QString> GetTableNames() = 0;

  /**
   * \brief Returns whether the given table should be scaled to the window and level.
   */
  virtual bool GetIsScaled(const QString& lookupTableName) = 0;

};

MITK_DECLARE_SERVICE_INTERFACE(QmitkLookupTableProviderService, "QmitkLookupTableProviderService/1.0")

#endif
