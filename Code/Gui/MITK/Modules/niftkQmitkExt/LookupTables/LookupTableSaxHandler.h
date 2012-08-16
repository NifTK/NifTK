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
#ifndef LOOKUPTABLESAXHANDLER_H
#define LOOKUPTABLESAXHANDLER_H

#include <niftkQmitkExtExports.h>
#include <vector>
#include <QString>
#include <QColor>
#include <QXmlDefaultHandler>

class vtkLookupTable;
class LookupTableContainer;

/**
 * \class LookupTableSaxHandler
 * \brief SAX handler to load lookup tables into LookupTableContainer objects.
 * \ingroup uk_ac_ucl_cmic_imagelookuptables_internal
 *
 * This class is not designed to be re-used. For each lookup table,
 * you create a new LookupTableSaxHander, parse the XML file, then call GetLookupTable()
 * to get the pointer to the created lookup table. The client then
 * must store and keep track of that pointer. This class must then be thrown away.
 *
 * This class is not thread safe, so you should load lookup tables
 * one at a time, in a single thread.
 */
class NIFTKQMITKEXT_EXPORT LookupTableSaxHandler : public QXmlDefaultHandler {

public:

	/** No-arg constructor. */
	LookupTableSaxHandler();

	/** Returns the internal lookup table, you should not call this until the parsing has finished sucessfully. */
	LookupTableContainer* GetLookupTableContainer();

	/** Methods that we must implement for the handler. */
  bool startElement(const QString &namespaceURI,
                    const QString &localName,
                    const QString &qName,
                    const QXmlAttributes &attributes);

  /** Methods that we must implement for the handler. */
  bool endElement(const QString &namespaceURI,
                  const QString &localName,
                  const QString &qName);

  /** Methods that we must implement for the handler. */
  bool characters(const QString &str);

  /** Methods that we must implement for the handler. */
  bool fatalError(const QXmlParseException &exception);

private:

  /** To temporarily store the premultiplied flag. */
  bool m_IsPreMultiplied;

  /** To temporarily store the order. */
  int m_Order;

  /** To temporarily store the display name. */
  QString m_DisplayName;

  /** We load all colours into a list first, so we know how long it is when we create vtkLookupTable. */
  std::vector<QColor> m_List;

};
#endif
