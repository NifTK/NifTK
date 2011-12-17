/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef LOOKUPTABLESAXHANDLER_CPP
#define LOOKUPTABLESAXHANDLER_CPP

#include <QMessageBox>

#include "LookupTableSaxHandler.h"
#include "LookupTableContainer.h"
#include "vtkLookupTable.h"
#include "mitkLogMacros.h"

LookupTableSaxHandler::LookupTableSaxHandler()
{
	m_IsPreMultiplied = false;
	m_Order = -1;
	m_DisplayName = QString("None");
	m_List.clear();
}

LookupTableContainer* LookupTableSaxHandler::GetLookupTableContainer()
{
  MITK_DEBUG << "GetLookupTableContainer():list.size()=" << m_List.size();

	vtkLookupTable *lookupTable = vtkLookupTable::New();
	lookupTable->SetRampToLinear();
	lookupTable->SetScaleToLinear();
	lookupTable->SetNumberOfTableValues(m_List.size());

	for (unsigned int i = 0; i < m_List.size(); i++)
	{
		QColor c = m_List[i];
		lookupTable->SetTableValue(i, c.redF(), c.greenF(), c.blueF(), c.alphaF());
	}

	LookupTableContainer *lookupTableContainer = new LookupTableContainer(lookupTable);
	lookupTableContainer->SetOrder(m_Order);
	lookupTableContainer->SetDisplayName(m_DisplayName);

	return lookupTableContainer;
}

bool LookupTableSaxHandler::startElement(
		                                     const QString & /* namespaceURI */,
                                         const QString & /* localName */,
                                         const QString &qName,
                                         const QXmlAttributes &attributes)
{
	if (qName == "lut")
	{

		m_Order = (attributes.value("order")).toInt();
		m_DisplayName = attributes.value("displayName");

		int premultiplied = (attributes.value("premultiplied")).toInt();
		if (premultiplied == 1)
		{
			m_IsPreMultiplied = true;
		}

	}
	else if (qName == "colour")
	{
		float red = (attributes.value("r")).toFloat();
		float green = (attributes.value("g")).toFloat();
		float blue = (attributes.value("b")).toFloat();

		if (!m_IsPreMultiplied)
		{
			red /= 255;
			green /= 255;
			blue /= 255;
		}

		QColor tmp;
		tmp.setRedF(red);
		tmp.setGreenF(green);
		tmp.setBlueF(blue);

		m_List.push_back(tmp);

	}
	else
	{
	  MITK_ERROR << "startElement():qName=" << qName.toLocal8Bit().constData() << ", which is unrecognised";
		return false;
	}
  return true;
}

bool LookupTableSaxHandler::characters(const QString &str)
{
    return true;
}

bool LookupTableSaxHandler::endElement(
                                       const QString & /* namespaceURI */,
                                       const QString & /* localName */,
                                       const QString &qName)
{
    return true;
}

bool LookupTableSaxHandler::fatalError(const QXmlParseException &exception)
{
    QMessageBox::warning(0, QObject::tr("SAX Handler"),
                         QObject::tr("Parse error at line %1, column "
                                     "%2:\n%3.")
                         .arg(exception.lineNumber())
                         .arg(exception.columnNumber())
                         .arg(exception.message()));
    return false;
}

#endif
