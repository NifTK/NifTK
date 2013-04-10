/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef LOOKUPTABLECONTAINER_H
#define LOOKUPTABLECONTAINER_H

#include <niftkCoreGuiExports.h>
#include <QString>
#include "vtkLookupTable.h"

/**
 * \class LookupTableContainer
 * \brief Class to contain a vtkLookupTable and to store meta-data attributes
 * like display name, which order to display it in in GUI, etc.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 *
 * Note that this container stores a const vtkLookupTable. Thus, the lookup table
 * is loaded from disc, when the application starts, and then clients should
 * copy it, and manage the copy themselves. You should never modify a lookup table
 * once it has been loaded, and you must never share them.
 */
class NIFTKCOREGUI_EXPORT LookupTableContainer {

public:

	/** Constructor that takes a lookup table. */
  LookupTableContainer(const vtkLookupTable* lut);

	/** Destructor. */
	virtual ~LookupTableContainer();

	/** Get the vtkLookupTable. */
	const vtkLookupTable* GetLookupTable() const { return m_LookupTable; }

	/** Set the order that determines which order this vtkLookupTable will be displayed in GUI. */
	void SetOrder(int i) { m_Order = i;}

	/** Get the order that this lookup table will be displayed in GUI. */
	int GetOrder() const { return m_Order; }

	/** Set the display name. */
	void SetDisplayName(const QString s) { m_DisplayName = s; }

	/** Get the display name. */
	QString GetDisplayName() const { return m_DisplayName; }

protected:

private:

  /** Deliberately prohibit copy constructor. */
	LookupTableContainer(const LookupTableContainer&) {};

	/** Deliberately prohibit assignment. */
	void operator=(const LookupTableContainer&) {};

	/** This is it! */
	const vtkLookupTable* m_LookupTable;

	/** Display name for display in GUI. */
	QString m_DisplayName;

	/** Store the order that it is to be displayed in GUI. */
	int m_Order;
};
#endif
