/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLookupTableManager_h
#define niftkLookupTableManager_h

#include <unordered_map>
#include <vector>

#include <QString>


class vtkLookupTable;

namespace niftk
{

class LookupTableContainer;

/**
 * \class LookupTableManager
 * \brief Class to manage access to LookupTableContainers.
 *
 * Each LookupTableContainers contains 1 vtkLookupTable.
 * These are loaded from disc when the LookupTableManager is created.
 * This manager class takes care of providing copies of the
 * lookup tables. So, when the client calls CloneLookupTable(),
 * the client owns the provided vtkLookupTable, and should delete it when done.
 */
class LookupTableManager
{

public:

  typedef std::unordered_map<std::string, const LookupTableContainer*> LookupTableMapType;

  /** No-arg constructor. */
  LookupTableManager();

  /** Destructor, to get rid of all lookup tables. */
  virtual ~LookupTableManager();

  /**
   * Gets the total number of lookup tables loaded.
   */
  unsigned int GetNumberOfLookupTables();

  /**
   * Returns a pointer to the nth lookup table container in the list, or NULL
   * if index < 0, or index >= GetNumberOfLookupTables().
   */
  const LookupTableContainer* GetLookupTableContainer(const QString& name);

  /**
   * \brief Returns the list of names in the map.
   */
  std::vector<QString> GetTableNames();

  /**
  * \brief Add the given LookupTableContainer to the set of containers
  */
  void AddLookupTableContainer(const LookupTableContainer* container);

  /**
  * \brief Replace the LookupTableContainer with name with the given LookupTableContainer
  */
  void ReplaceLookupTableContainer(const LookupTableContainer* container, const QString& name);

  /** Checks that name exists within the containers map. */
  bool CheckName(const QString& name);

private:

  /** A list of lookup table containers that we have loaded. */
  LookupTableMapType m_Containers;

};

}

#endif
