/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASSingleViewWidgetListManager.h"

#include <mitkDataNode.h>
#include <QmitkRenderWindow.h>
#include "QmitkMIDASSingleViewWidget.h"

QmitkMIDASSingleViewWidgetListManager::QmitkMIDASSingleViewWidgetListManager()
{
  m_DataNodes.clear();
  m_Widgets.clear();
}

QmitkMIDASSingleViewWidgetListManager::~QmitkMIDASSingleViewWidgetListManager()
{
}

void QmitkMIDASSingleViewWidgetListManager::RegisterWidget(QmitkMIDASSingleViewWidget *widget)
{
  std::set<mitk::DataNode*> newNodes;
  m_DataNodes.push_back(newNodes);
  m_Widgets.push_back(widget);
}

void QmitkMIDASSingleViewWidgetListManager::DeRegisterAllWidgets()
{
  this->DeRegisterWidgets(0, m_Widgets.size()-1);
}

void QmitkMIDASSingleViewWidgetListManager::DeRegisterWidgets(unsigned int startWindowIndex, unsigned int endWindowIndex)
{
  m_DataNodes.erase(m_DataNodes.begin() + startWindowIndex, m_DataNodes.begin() + endWindowIndex+1);
  m_Widgets.erase(m_Widgets.begin() + startWindowIndex, m_Widgets.begin() + endWindowIndex+1);
}

int QmitkMIDASSingleViewWidgetListManager::GetIndexFromWindow(QmitkRenderWindow* window)
{
  int result = -1;

  for (unsigned int i = 0; i < m_Widgets.size(); i++)
  {
    bool contains = m_Widgets[i]->ContainsWindow(window);
    if (contains)
    {
      result = i;
      break;
    }
  }
  return result;
}

int QmitkMIDASSingleViewWidgetListManager::GetNumberOfNodesRegisteredWithWidget(int windowIndex)
{
  int result = m_DataNodes[windowIndex].size();
  return result;
}

void QmitkMIDASSingleViewWidgetListManager::RemoveNode( const mitk::DataNode* node)
{
  for (unsigned int i = 0; i < m_DataNodes.size(); i++)
  {
    std::set<mitk::DataNode*>::iterator iter;
    iter = m_DataNodes[i].find(const_cast<mitk::DataNode*>(node));
    if (iter != m_DataNodes[i].end())
    {
      m_DataNodes[i].erase(iter);
    }
  }
}
