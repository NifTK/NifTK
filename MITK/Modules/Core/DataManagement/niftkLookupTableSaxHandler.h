/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLookupTableSaxHandler_h
#define niftkLookupTableSaxHandler_h

#include <niftkCoreExports.h>

#include <vector>

#include <QColor>
#include <QString>
#include <QXmlDefaultHandler>

class vtkLookupTable;

namespace niftk
{

class LookupTableContainer;

/**
 * \class LookupTableSaxHandler
 * \brief SAX handler to load lookup tables into LookupTableContainer objects.
 *
 * This class is not designed to be re-used. For each lookup table,
 * you create a new LookupTableSaxHander, parse the XML file, then call GetLookupTable()
 * to get the pointer to the created lookup table. The client then
 * must store and keep track of that pointer. This class must then be thrown away.
 *
 * This class is not thread safe, so you should load lookup tables
 * one at a time, in a single thread.
 */
class NIFTKCORE_EXPORT LookupTableSaxHandler : public QXmlDefaultHandler
{

public:

  /** No-arg constructor. */
  LookupTableSaxHandler();

  /** Returns the internal lookup table, you should not call this until the parsing has finished sucessfully. */
  LookupTableContainer* GetLookupTableContainer();

  /** Methods that we must implement for the handler. */
  bool startElement(const QString& namespaceURI,
                    const QString& localName,
                    const QString& qName,
                    const QXmlAttributes& attributes) override;

  /** Methods that we must implement for the handler. */
  bool endElement(const QString& namespaceURI,
                  const QString& localName,
                  const QString& qName) override;

  /** Methods that we must implement for the handler. */
  bool characters(const QString& str) override;

  /** Methods that we must implement for the handler. */
  bool fatalError(const QXmlParseException& exception) override;

private:

  /** To temporarily store the premultiplied flag. */
  bool m_IsPreMultiplied;

  /** To temporarily store the order. */
  int m_Order;

  /** To temporarily store the scaled flag. */
  bool m_IsScaled;

  /** To temporarily store the display name. */
  QString m_DisplayName;

  /** We load all colours into a list first, so we know how long it is when we create vtkLookupTable. */
  std::vector<QColor> m_List;

};

}

#endif
