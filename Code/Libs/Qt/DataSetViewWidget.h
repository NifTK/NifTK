/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-24 12:19:23 +0100 (Sun, 24 Jul 2011) $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef DATASETVIEWWIDGET_H
#define DATASETVIEWWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include <QFrame>
#include <QVector>
#include <QStringList>

/**
 * \class DataSetViewWidget
 * \brief Composite widget to provide a list of datasets, that can be toggled for visibility and re-ordered.
 */

class DataSetViewListWidget;
class QListWidgetItem;
class QPushButton;
class QGridLayout;

class NIFTKQT_WINEXPORT DataSetViewWidget : public QFrame
{
    Q_OBJECT

public:

  /** Constructor, calls setupUi to create widgets.  */
  DataSetViewWidget(QWidget* parent = 0);

  /** Destructor. */
  ~DataSetViewWidget();

  /** Toggles the buttons visibility, as some screens won't need them. */
  void SetButtonsVisible(bool b);

  /** Returns a pointer to the Up button. */
  QPushButton* GetUpButton() const { return m_UpButton; }

	/** Returns a pointer to the Down button. */
	QPushButton* GetDownButton() const { return m_DownButton; }

	/** Returns a copy of the currently selected label(s) (eg. Filename(s)). */
	QStringList GetCurrentlySelectedLabels();

	/** Returns all the labels. */
	QStringList GetAllLabels();

	/** Returns true if the given string is currently in the viewer. */
	bool IsStringInList(QString name) const;

	/** Works out the index from the filename. */
	int GetIndex(QString name) const;

	/** Sets the currently selected index. */
	void SetIndex(int i);

	/** Sets the widget so that the give filename is selected. */
	void SetSelectedFileName(QString name);

	/** If b==true, Up and Down buttons are Enabled, else Disabled. */
	void SetEnableButtons(bool b);

public slots:

	/** When the up button is pressed we move an item up in the list. */
	void UpButtonPressed();

	/** When the down button is pressed we move an item down in the list. */
	void DownButtonPressed();

	/** This would occur, if the view had focus, and the user moved cursor up or down, and additionally when user clicks on item thats not the current item */
	void SelectionChanged();

	/** This would occur when the user double clicks an item, which we use to toggle visibility. */
	void ItemDoubleClicked(QListWidgetItem* current);

  /** Call this to indicate to the widget that dataset/s are being loaded. The identifier string is a list of filenames. */
	void DatasetLoaded(QStringList identifierList);

	/** Call this to indicate to the widget that multiple datasets have been unloaded. The identifier string is a list of filenames. */
	void DatasetUnLoaded(QStringList identifier);

signals:

	/** Indicates that an item's order was switched. */
	void ItemsSwitched(QString itemAName, QString itemBName, int indexA, int indexB);

	/** Indicates that an item was selected. */
	void ItemsSelected(QStringList oldNames, QStringList newNames, QVector<int> oldIndexes, QVector<int> newIndexes);

	/** Indicates that an item's visibility was toggled. */
	void ItemVisibilityToggled(int currentIndex, bool currentValue);

protected:

private:

	DataSetViewWidget(const DataSetViewWidget&);  // Not implemented.
	void operator=(const DataSetViewWidget&);  // Not implemented.

	/** Actually creates the GUI widgets. */
	void SetupUi(QWidget *parent);

	/** Works out the row number from the widget item. */
	int FindRow(QListWidgetItem* current);

	/** Function to swap items in the list around. */
	void SwitchItems(int indexA, int indexB);

	DataSetViewListWidget *m_ListWidget;
	QPushButton *m_UpButton;
	QPushButton *m_DownButton;
	QGridLayout *m_Layout;

	QVector<bool> m_Visibility;
	QStringList m_PreviousSelectionText;
	QStringList m_CurrentSelectionText;
	QVector<int> m_PreviousSelectionIndex;
	QVector<int> m_CurrentSelectionIndex;
};
#endif
