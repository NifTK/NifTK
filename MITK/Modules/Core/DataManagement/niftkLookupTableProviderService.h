/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLookupTableProviderService_h
#define niftkLookupTableProviderService_h

#include <string>
#include <vector>

// Microservices
#include <mitkServiceInterface.h>

#include <niftkLabeledLookupTableProperty.h>
#include <niftkNamedLookupTableProperty.h>

class vtkLookupTable;

namespace niftk
{

class LookupTableContainer;

/**
 * \class LookupTableProviderService
 * \brief Service to provide lookup tables.
 */
struct LookupTableProviderService
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
   * \brief Same as CreateLookupTable, but wraps it into a niftk::NamedLookupTableProperty.
   */
  virtual NamedLookupTableProperty::Pointer CreateLookupTableProperty(const QString& lookupTableName,
                                                                            float lowestValueOpacity,
                                                                            float highestValueOpacity) = 0;

  /**
   * \brief Similar to CreateLookupTable, but create a niftk::LabeledLookupTableProperty.
   */
  virtual LabeledLookupTableProperty::Pointer CreateLookupTableProperty(const QString& lookupTableName) = 0;

  /**
   * \brief Add a new LookupTableContainer to the LookupTableManager.
   */
  virtual void AddNewLookupTableContainer(const LookupTableContainer* container) = 0;

  /**
   * \brief Replace the LookupTableContainer of the given name with another.
   */
  virtual void ReplaceLookupTableContainer(const LookupTableContainer* container, const QString& lookupTableName) = 0;

  /**
   * \brief Returns the display names of  all table.
   */
  virtual std::vector<QString> GetTableNames() = 0;

  /**
   * \brief Returns whether the given table should be scaled to the window and level.
   */
  virtual bool GetIsScaled(const QString& lookupTableName) = 0;

  /// \brief Attempts to load LookupTable from given file, returning display name of LookupTable if successful.
  virtual QString LoadLookupTable(const QString& fileName) = 0;

};

}

MITK_DECLARE_SERVICE_INTERFACE(niftk::LookupTableProviderService, "LookupTableProviderService/1.0")

#endif
