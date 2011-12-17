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
#ifndef QTFRAMEDVIEW_H
#define QTFRAMEDVIEW_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include <QFrame>

/**
 * \class QtFramedView
 * \brief Class to provide simple methods to manipulate the borders of a QFrame widget.
 */
class NIFTKQT_WINEXPORT QtFramedView : public QFrame
{

	Q_OBJECT

public:

	/** Destructor. */
	~QtFramedView();

	/** Constructor. */
	QtFramedView(QWidget *parent = 0);

	/** Sets the frame to red. */
	void SetFrameToRed();

  /** Sets the frame to white. */
  void SetFrameToWhite();

  /** Sets the frame to black. */
  void SetFrameToBlack();

  /** Sets the frame to yellow. */
  void SetFrameToYellow();

  /** Sets the frame to green. */
  void SetFrameToGreen();

  /** Sets the frame to blue. */
  void SetFrameToBlue();

	/** If b is true, we display a selected look (red), else, we display an unselected look (white). */
	void SetSelectedLook(bool b);

	/** Returns true if this widget is selected, and false otherwise. */
	bool IsSelected() const { return m_IsSelected; }

	/** In Qt parlance, if b==true, we draw a Plane frame, if b==False we draw a NoFrame. */
	void SetFrameVisible(bool b);

	/** Returns whether the frame should be visible. This is different to widget visibility. */
	bool IsFrameVisible() const { return m_IsVisible; }

public slots:

signals:

protected:

private:

	/** Deliberately prohibit copy constructor. */
  QtFramedView(const QtFramedView&){};

	/** Deliberately prohibit assignment. */
	void operator=(const QtFramedView&){};

	/** Returns whether this thing is selected. */
	bool m_IsSelected;

	/** Returns whether the frame should currently be on (Plane frame), or off (No Frame). */
	bool m_IsVisible;

};
#endif
