/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkLookupTableContainer_h
#define QmitkLookupTableContainer_h

#include <niftkCoreGuiExports.h>

#include <QString>
#include <vtkLookupTable.h>

/**
 * \class QmitkLookupTableContainer
 * \brief Class to contain a vtkLookupTable and to store meta-data attributes
 * like display name, which order to display it in in GUI, etc.
 */
class NIFTKCOREGUI_EXPORT QmitkLookupTableContainer {

public:

  /** Constructor that takes a lookup table. */
  QmitkLookupTableContainer(const vtkLookupTable* lut);

  /** Destructor. */
  virtual ~QmitkLookupTableContainer();

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

  /** Set scaled property. */
  void SetIsScaled(const bool s) { m_IsScaled = s; }

  /** Get scaled property. */
  bool GetIsScaled() const { return m_IsScaled; }

  typedef std::pair<int, std::string> LabelType;
  typedef std::vector<LabelType> LabelsListType;  

  /** Set labels. */
  void SetLabels(LabelsListType labels){ m_Labels = labels; }
  
  /** Get labels. */
 LabelsListType GetLabels()const { return m_Labels; }

protected:

private:

  /** Deliberately prohibit copy constructor. */
  QmitkLookupTableContainer(const QmitkLookupTableContainer&) {}

  /** Deliberately prohibit assignment. */
  void operator=(const QmitkLookupTableContainer&) {}

  /** This is it! */
  const vtkLookupTable* m_LookupTable;

  /** Display name for display in GUI. */
  QString m_DisplayName;

  /** What type of lookup table */
  bool m_IsScaled;

  /** Store the order that it is to be displayed in GUI. */
  int m_Order;

  /** Labels for the entries in the vtkLUT (optional)*/
  LabelsListType m_Labels;

};
#endif
