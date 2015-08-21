/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkLookupTableManager_h
#define QmitkLookupTableManager_h

#include <niftkCoreGuiExports.h>
#include <unordered_map>
#include <QString>

/**
 * \class QmitkLookupTableManager
 * \brief Class to manage access to QmitkLookupTableContainers.
 *
 * Each QmitkLookupTableContainers contains 1 vtkLookupTable.
 * These are loaded from disc when the QmitkLookupTableManager is created.
 * This manager class takes care of providing copies of the
 * lookup tables. So, when the client calls CloneLookupTable(),
 * the client owns the provided vtkLookupTable, and should delete it when done.
 */

class QmitkLookupTableContainer;
class vtkLookupTable;

class NIFTKCOREGUI_EXPORT QmitkLookupTableManager {

public:

  typedef std::unordered_map<std::string, const QmitkLookupTableContainer*> LookupTableMapType;

  /** No-arg constructor. */
  QmitkLookupTableManager();

  /** Destructor, to get rid of all lookup tables. */
  virtual ~QmitkLookupTableManager();

  /**
   * Gets the total number of lookup tables loaded.
   */
  unsigned int GetNumberOfLookupTables();

  /**
   * Returns a pointer to the nth lookup table container in the list, or NULL
   * if index < 0, or index >= GetNumberOfLookupTables().
   */
  const QmitkLookupTableContainer* GetLookupTableContainer(QString& name);

  /**
   * \brief Returns the list of names in the map.
   */
  std::vector<QString> GetTableNames();  
  
  /**
  * \brief Add the given LookupTableContainer to the set of containers
  */
  void AddLookupTableContainer(QmitkLookupTableContainer* container);

  /**
  * \brief Replace the LookupTableContainer with name with the given LookupTableContainer
  */
  void ReplaceLookupTableContainer(QmitkLookupTableContainer* container, QString& name);

  /** Checks that name exists within the containers map. */
  bool CheckName(QString& name);

private:

  /** A list of lookup table containers that we have loaded. */
  LookupTableMapType m_Containers;

};
#endif
