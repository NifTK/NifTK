/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef LOOKUPTABLEMANAGER_H
#define LOOKUPTABLEMANAGER_H

#include <vector>

/**
 * \class LookupTableManager
 * \brief Class to manage access to LookupTableContainers.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 *
 * Each LookupTableContainer contains 1 vtkLookupTable.
 * These are loaded from disc when the LookupTableManager is created.
 * This manager class takes care of providing copies of the
 * lookup tables. So, when the client calls CloneLookupTable(),
 * the client own the provided vtkLookupTable, and should delete it when done.
 */

class LookupTableContainer;
class vtkLookupTable;

class LookupTableManager {

public:

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
  const LookupTableContainer* GetLookupTableContainer(const unsigned int& n);

  /**
   * Returns a copy of the nth lookup table in the list, or NULL
   * if index < 0, or index >= GetNumberOfLookupTables().
   */
  vtkLookupTable* CloneLookupTable(const unsigned int& n);

private:

  /** A list of lookup table containers that we have loaded. */
  std::vector<const LookupTableContainer*> m_List;

  /** Checks that the index is within range. */
  bool CheckIndex(const unsigned int& n);

};
#endif
