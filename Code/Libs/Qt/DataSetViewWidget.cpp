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
#ifndef DATASETVIEWWIDGET_CPP
#define DATASETVIEWWIDGET_CPP
#include "DataSetViewWidget.h"
#include "DataSetViewListWidget.h"
#include <Qt>
#include <QPushButton>
#include <QListWidgetItem>
#include <QModelIndex>
#include <QMouseEvent>
#include <QApplication>
#include <iostream>
#include <QGridLayout>

DataSetViewWidget::DataSetViewWidget(QWidget* parent)
: m_UpButton(NULL),
  m_DownButton(NULL)
{
  this->m_Visibility.clear();
  this->setParent(parent);
	this->SetupUi(this);
}

DataSetViewWidget::~DataSetViewWidget()
{
	// Don't need to delete Qt widgets that are created with new ClassName(*parentWidget)
}

void DataSetViewWidget::SetupUi(QWidget *parent)
{
  m_ListWidget = new DataSetViewListWidget(parent);
  m_ListWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);

	m_UpButton = new QPushButton(parent);
	m_UpButton->setText(tr("up"));

	m_DownButton = new QPushButton(parent);
	m_DownButton->setText(tr("down"));

	m_Layout = new QGridLayout(this);
	m_Layout->addWidget(m_ListWidget, 0, 0, 1, 2);
	m_Layout->addWidget(m_UpButton, 1, 0);
	m_Layout->addWidget(m_DownButton, 1, 1);

  // Ticket 541
	m_ListWidget->setDragEnabled(true);
  this->SetButtonsVisible(true);

  connect(m_UpButton, SIGNAL(pressed()), this, SLOT(UpButtonPressed()));
  connect(m_DownButton, SIGNAL(pressed()), this, SLOT(DownButtonPressed()));
  connect(m_ListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(SelectionChanged()));

  // Not using this at the moment, and also we probably wont.
  // In our viewer you can toggle between images, and drop any number of images into each view, so
  // probably don't need to toggle visibility by double clicking like you do in FSLView.
  //connect(m_ListWidget, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(ItemDoubleClicked(QListWidgetItem*)));
}

void DataSetViewWidget::SwitchItems(int currentIndex, int nextIndex)
{
  // Swap visibility flags round.
  bool tmp = m_Visibility[currentIndex];
  m_Visibility[currentIndex] = m_Visibility[nextIndex];
  m_Visibility[nextIndex] = tmp;

  // Swap items in widget round
  QListWidgetItem *item = m_ListWidget->takeItem(nextIndex);
  m_ListWidget->blockSignals(true);
  m_ListWidget->insertItem(currentIndex, item);
  m_ListWidget->setCurrentRow(nextIndex);
  m_ListWidget->blockSignals(false);

  if (!this->signalsBlocked())
  {
    emit ItemsSwitched(m_ListWidget->item(currentIndex)->text(), m_ListWidget->item(nextIndex)->text(), currentIndex, nextIndex);
  }
}

void DataSetViewWidget::UpButtonPressed()
{
	int currentIndex = m_ListWidget->currentRow();

	if (currentIndex > 0)
	{
		int nextIndex = currentIndex - 1;
		this->SwitchItems(currentIndex, nextIndex);
	}
}

void DataSetViewWidget::DownButtonPressed()
{
	int currentIndex = m_ListWidget->currentRow();

	if (currentIndex < m_ListWidget->count() - 1)
	{
		int nextIndex = currentIndex + 1;
		this->SwitchItems(currentIndex, nextIndex);
	}
}

int DataSetViewWidget::FindRow(QListWidgetItem* current)
{
	for (int i = 0; i < m_ListWidget->count(); i++)
	{
		if (m_ListWidget->item(i) == current)
		{
			return i;
		}
	}
	return -1;
}

void DataSetViewWidget::SelectionChanged()
{
  QList<QListWidgetItem *> selectedItems = m_ListWidget->selectedItems();

  m_CurrentSelectionText.clear();
  m_CurrentSelectionIndex.clear();

  for (int i = 0; i < selectedItems.size(); i++)
  {
    QString itemText = selectedItems.at(i)->text();
    int indexOfText = this->GetIndex(itemText);

    m_CurrentSelectionText.push_back(itemText);
    m_CurrentSelectionIndex.push_back(indexOfText);
  }

  if (!this->signalsBlocked())
  {
    emit ItemsSelected(m_PreviousSelectionText, m_CurrentSelectionText, m_PreviousSelectionIndex, m_CurrentSelectionIndex);
  }

  m_PreviousSelectionText.clear();
  m_PreviousSelectionIndex.clear();

  for (int i = 0; i < m_CurrentSelectionText.size(); i++)
  {
    m_PreviousSelectionText.push_back(m_CurrentSelectionText[i]);
    m_PreviousSelectionIndex.push_back(m_CurrentSelectionIndex[i]);
  }
}

void DataSetViewWidget::ItemDoubleClicked(QListWidgetItem* current)
{
	int currentIndex = this->FindRow(current);

	if (currentIndex >= 0)
	{
		m_Visibility[currentIndex] = !m_Visibility[currentIndex];

		if (m_Visibility[currentIndex])
		{
			current->setIcon(QIcon(":/images/eye.png"));
		}
		else
		{
			current->setIcon(QIcon(":/images/blank.png"));
		}

	  if (!this->signalsBlocked())
	  {
		  emit ItemVisibilityToggled(currentIndex, m_Visibility[currentIndex]);
	  }
	}
}



/** Call this to indicate to the widget that dataset/s are being loaded. 
The identifier string is a list of filenames. */
void DataSetViewWidget::DatasetLoaded(QStringList identifierList)
{
  for(int i = 0; i < identifierList.size(); i++)
  {
    QListWidgetItem *item = new QListWidgetItem(QIcon(":/images/eye.png"), identifierList.at(i), m_ListWidget);
    m_ListWidget->addItem(item);
    m_ListWidget->repaint();
	  m_Visibility.append(true);
  }

	if (m_ListWidget->count() >= 1)
	{
	  m_ListWidget->setCurrentItem(m_ListWidget->item(0), QItemSelectionModel::SelectCurrent);
	}

	// Without this line, you need to click on the scrollbar before
	// the new filename/s becomes visible. This forces the internal viewport to update.
	m_ListWidget->viewport()->repaint();
}


/** Call this when a dataset has been unloaded. 
The identifier string is what should appear in widget, i.e. most likely a filename. */
void DataSetViewWidget::DatasetUnLoaded(QStringList identifierList)
{
  int itemRowNo = 0;
  for(int i = 0; i < identifierList.size(); i++)
  {
    QList<QListWidgetItem*> itemList = m_ListWidget->findItems(identifierList.at(i), Qt::MatchExactly);

    if(!itemList.empty())
      itemRowNo = m_ListWidget->row(itemList.at(0));

    m_Visibility.remove(itemRowNo);
    m_ListWidget->blockSignals(true);
  	QListWidgetItem* item = m_ListWidget->takeItem(itemRowNo);
  	m_ListWidget->blockSignals(false);
		delete item;
		m_ListWidget->repaint();
	}

}

QStringList DataSetViewWidget::GetCurrentlySelectedLabels()
{
  return m_ListWidget->GetCurrentlySelectedLabels();
}

QStringList DataSetViewWidget::GetAllLabels()
{
  return m_ListWidget->GetAllLabels();
}

void DataSetViewWidget::SetButtonsVisible(bool b)
{
  this->m_UpButton->setVisible(b);
  this->m_DownButton->setVisible(b);
}

int DataSetViewWidget::GetIndex(QString name) const
{
  int result = -1;
  for (int i = 0; i < m_ListWidget->count(); i++)
  {
    QListWidgetItem* item = m_ListWidget->item(i);

    if (item->text() == name)
    {
      result = i;
      break;
    }
  }

  return result;
}

void DataSetViewWidget::SetIndex(int i)
{
  m_ListWidget->blockSignals(true);
  m_ListWidget->setCurrentRow(i, QItemSelectionModel::Clear); // this was so that if you had selected multiple rows, you clear them
  m_ListWidget->setCurrentRow(i, QItemSelectionModel::Select); // and then this selects only the single item you specify.
  m_ListWidget->viewport()->repaint();
  m_ListWidget->blockSignals(false);
}

void DataSetViewWidget::SetSelectedFileName(QString name)
{
  int itemIndex = this->GetIndex(name);
  if (itemIndex != -1)
  {
    this->SetIndex(itemIndex);
  }
}

bool DataSetViewWidget::IsStringInList(QString name) const
{
  bool result = false;

  int itemIndex = this->GetIndex(name);
  if (itemIndex != -1)
  {
    result = true;
  }
  return result;
}

void DataSetViewWidget::SetEnableButtons(bool b)
{
  this->m_UpButton->setEnabled(b);
  this->m_DownButton->setEnabled(b);
}
#endif
