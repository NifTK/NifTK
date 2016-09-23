/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkBaseApplication.h>

#include <Poco/Util/HelpFormatter.h>
#include <Poco/Util/Option.h>

#include <QVariant>


namespace niftk
{

class NiftyMIDAS : public BaseApplication
{
public:
  static const QString PROP_VIEWER_NUMBER;
  static const QString PROP_DRAG_AND_DROP;
  static const QString PROP_WINDOW_LAYOUT;
  static const QString PROP_BIND_WINDOWS;
  static const QString PROP_BIND_VIEWERS;
  static const QString PROP_ANNOTATION;

  NiftyMIDAS(int argc, char **argv)
    : BaseApplication(argc, argv)
  {
    this->setApplicationName("NiftyMIDAS");
    this->setProperty(mitk::BaseApplication::PROP_APPLICATION, "uk.ac.ucl.cmic.niftymidas");
  }

  /// \brief Define command line arguments
  /// \param options
  void defineOptions(Poco::Util::OptionSet& options) override
  {
    BaseApplication::defineOptions(options);

    Poco::Util::Option viewersOption("viewer-number", "v",
        "\n"
        "Sets the number of viewers.\n"
        "\n"
        "Viewers can be arranged in rows and columns. By default (without this option),\n"
        "there is one viewer. You can specify the required rows and numbers in the form\n"
        "<rows>x<columns>. The row number can be omitted. If only one number is given, the\n"
        "viewers will appear side-by-side.\n");
    viewersOption.argument("[<rows>x]<columns>").binding(PROP_VIEWER_NUMBER.toStdString());
    options.addOption(viewersOption);

    Poco::Util::Option dragAndDropOption("drag-and-drop", "D",
        "\n"
        "Loads the data nodes into the given viewers.\n"
        "\n"
        "By default (without this option), the data nodes are added to the Data Manager,\n"
        "but not loaded into any viewer.\n"
        "\n"
        "<data nodes> is a comma separated list of data node names, as given with the\n"
        "'--open' option. At least one node has to be given.\n"
        "\n"
        "<viewers> is a comma separated list of viewer indices. Viewer indices are\n"
        "integer numbers starting from 1 and increasing row-wise. For instance, with\n"
        "2x3 viewers the index of the first viewer of the second row is 4. If no viewer\n"
        "index is given, the data nodes will be loaded into each viewer.\n");
    dragAndDropOption.argument("<data nodes>[:<viewers>]").repeatable(true);
    dragAndDropOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &NiftyMIDAS::HandleRepeatableOption));
    options.addOption(dragAndDropOption);

    Poco::Util::Option windowLayoutOption("window-layout", "l",
        "\n"
        "Sets the window layout of the given viewers.\n"
        "\n"
        "The window layout tells which windows of a viewer are visible and in which\n"
        "arrangement."
        "\n"
        "<viewers> is a comma separated list of viewer indices. Viewer indices are\n"
        "integer numbers starting from 1 and increasing row-wise. For instance, with\n"
        "2x3 viewers the index of the first viewer of the second row is 4. If no viewer\n"
        "index is given, the layout will be applied to each viewer.\n"
        "\n"
        "Valid layout names are the followings: 'axial', 'sagittal', 'coronal', 'as acquired'\n"
        "(the original orientation as the displayed image was acquired), '3D', '2x2'\n"
        "(coronal, sagittal, axial and 3D windows in a 2x2 arrangement), '3H' (coronal,\n"
        "sagittal and axial windows in horizontal arrangement), '3V' (sagittal, coronal\n"
        "and axial windows in vertical arrangement), 'cor sag H', 'cor sag V', 'cor ax H',\n"
        "'cor ax V', 'sag ax H' and 'sag ax V' (2D windows of the given orientation in\n"
        "horizontal or vertical arrangement, respectively).\n");
    windowLayoutOption.argument("[<viewers>:]<layout>").repeatable(true);
    windowLayoutOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &NiftyMIDAS::HandleRepeatableOption));
    options.addOption(windowLayoutOption);

    Poco::Util::Option bindWindowsOption("bind-windows", "B",
        "\n"
        "Sets the window binding mode of the given viewers.\n"
        "\n"
        "The window binding mode tells if the cursor position and the magnification is bound\n"
        "across the 2D windows of a viewer. It is relevant only for window layouts with multiple\n"
        "windows. If the cursor position is bound, the cursors (aka. crosshairs) of the windows\n"
        "will be aligned, and panning the images in one window will move them in the other windows\n"
        "accordingly, so that the cursors stay aligned. Similarly, if the magnification is bound,\n"
        "the scale factor will be the same in every 2D window of the layout, and zooming in one\n"
        "window will zoom the other windows, accordingly.\n"
        "\n"
        "<viewers> is a comma separated list of viewer indices. Viewer indices are\n"
        "integer numbers starting from 1 and increasing row-wise. For instance, with\n"
        "2x3 viewers the index of the first viewer of the second row is 4. If no viewer\n"
        "index is given, the window binding mode will be applied to each viewer.\n"
        "\n"
        "<binding mode options> is a comma separated list of window binding mode options,\n"
        "given in the following form:\n"
        "\n"
        "    <binding mode name>[=<binding mode value>].\n"
        "\n"
        "Valid binding mode names are 'cursor', 'magnification' and 'all'. Valid binding\n"
        "mode values are 'true', 'on', 'yes', 'false', 'off' and 'no'. Window bindings are\n"
        "enabled by default. If you want to disable the window bindings for a viewer\n"
        "completely (both cursor and magnification), you can specify 'all=off' as '<binding\n"
        "mode options>' without naming the individual binding modes.\n");
    bindWindowsOption.argument("[<viewers>:]<binding mode options>").repeatable(true);
    bindWindowsOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &NiftyMIDAS::HandleRepeatableOption));
    options.addOption(bindWindowsOption);

    Poco::Util::Option bindViewersOption("bind-viewers", "b",
        "\n"
        "Sets the binding mode across viewers.\n"
        "\n"
        "The binding mode tells if the selected position, cursor position, magnification,\n"
        "window layout and world geometry is bound across viewers. The viewer binding mode is\n"
        "relevant only with multiple viewers. If the selected position is bound, the cursors\n"
        "will indicate the same world (mm) position in each viewer. If the cursor position is\n"
        "bound, the cursors (aka. crosshairs) of the viewers will be aligned, and panning the\n"
        "images in one viewer will move them in the other viewers accordingly, so that the\n"
        "cursors stay aligned. Similarly, if the magnification is bound, the scale factor will\n"
        "be the same in the viewers, and zooming in one window will zoom in the seleced window\n"
        "of the other viewers, accordingly. Cursor and magnification binding binds the selected\n"
        "2D windows of viewers, not every window. (You can use it in combination with window\n"
        "bindings.)\n"
        "\n"
        "<viewers> is a comma separated list of viewer indices. Viewer indices are\n"
        "integer numbers starting from 1 and increasing row-wise. For instance, with\n"
        "2x3 viewers the index of the first viewer of the second row is 4. If no viewer\n"
        "index is given, the window binding mode will be applied to each viewer.\n"
        "\n"
        "<binding mode options> is a comma separated list of window binding mode options,\n"
        "given in the following form:\n"
        "\n"
        "    <binding mode name>[=<binding mode value>].\n"
        "\n"
        "Valid binding mode names are 'position', 'cursor', 'magnification', 'layout', 'geometry'\n"
        "and 'all'. Valid binding mode values are 'true', 'on', 'yes', 'false', 'off' and 'no'.\n"
        "Viewer bindings are disabled by default. If you want to enable every kinds of bindings\n"
        "for viewers, you can specify 'all=on' as '<binding mode options>' without naming the\n"
        "individual binding modes.\n");
    bindViewersOption.argument("[<viewers>:]<binding mode options>").binding(PROP_BIND_VIEWERS.toStdString());
    options.addOption(bindViewersOption);

    Poco::Util::Option annotationOption("annotation", "a",
        "\n"
        "Sets a list of properties to be displayed as annotations in the viewers.\n"
        "\n"
        "<viewers> is a comma separated list of viewer indices. Viewer indices are "
        "integer numbers starting from 1 and increasing row-wise. For instance, with "
        "2x3 viewers the index of the first viewer of the second row is 4. If no viewer "
        "index is given, the annotations will displayed in each viewer.\n"
        "\n"
        "<properties> is a comma separated list of data node property names. The property "
        "values will be displayed in the bottom left corner of the selected render window "
        "of the given viewers. The values will be shown for images that are visible in "
        "those viewers.\n");
    annotationOption.argument("[<viewers>:]<properties>").repeatable(true);
    annotationOption.callback(Poco::Util::OptionCallback<BaseApplication>(this, &NiftyMIDAS::HandleRepeatableOption));
    options.addOption(annotationOption);
  }

  virtual Poco::Util::HelpFormatter* CreateHelpFormatter() override
  {
    Poco::Util::HelpFormatter* helpFormatter = BaseApplication::CreateHelpFormatter();

    std::string footer = helpFormatter->getFooter();

    QString examples =
        "\n"
        "The following command opens two reference images and one segmentation, and opens "
        "them in two viewers, side-by-side:\n"
        "\n"
        "    ${commandName} -o mask:segmentation.nii -o timeline1:image1.nii -o timeline2:image2.nii \\\n"
        "        --viewer-number 2 --drag-and-drop timeline1,mask:1 --drag-and-drop timeline2,mask:2\n"
        "\n"
        "Note that 'mask' has to be specified before the reference images so that it is "
        "placed in an upper layer than the reference images.\n"
        "\n"
        "The following command opens an image, loads it to the viewer, and switches the "
        "window layout to the sagittal window:\n"
        "\n"
        "    ${commandName} -o T1:image1.nii -D T1 --window-layout sagittal\n"
        "\n"
        "The following command opens two images in separate viewers, and binds the cursor "
        "positions and the magnification of the viewers:\n"
        "\n"
        "    ${commandName} -o timeline1:image1.nii -o timeline2:image2.nii -v 2 \\\n"
        "        -D timeline1:1 -D timeline2:2 --bind-viewers position,cursor,magnification=on\n"
        "\n"
        "The following command opens two images in separate viewers, and enables all "
        "kinds of bindings across the viewers:\n"
        "\n"
        "    ${commandName} -o timeline1:image1.nii -o timeline2:image2.nii -v 2 \\\n"
        "        -D timeline1:1 -D timeline2:2 -b all=on\n"
        "\n"
        "The following command opens an image, loads it into the viewer, sets the 2x2 window "
        "layout and switches off the cursor and magnification binding across the 2D windows:\n"
        "\n"
        "    ${commandName} -o T1:image1.nii -D T1 -l 2x2 --bind-windows all:off\n"
        "\n"
        "The following command opens two images, loads each to a different viewer, sets the "
        "patient name and the aquisition time properties and displays them as annotations "
        "in both viewers:\n"
        "\n"
        "    ${commandName} -o A:image1.nii -o B:image2.nii -v 2 -D A:1 -D B:2 \\\n"
        "        -p A:Patient=\"John Doe\" -p B:Patient=\"Jane Doe\" \\\n"
        "        -p A:Acquired=\"2016.06.06 15:05\" -p B:Acquired=\"2016.07.07 10:35\" \\\n"
        "        --annotation Patient,Aquired\n"
        "";

    examples.replace("${commandName}", QString::fromStdString(this->commandName()));

    footer += examples.toStdString();

    helpFormatter->setFooter(footer);

    return helpFormatter;
  }

};

const QString NiftyMIDAS::PROP_VIEWER_NUMBER = "applicationArgs.viewer-number";
const QString NiftyMIDAS::PROP_DRAG_AND_DROP = "applicationArgs.drag-and-drop";
const QString NiftyMIDAS::PROP_WINDOW_LAYOUT = "applicationArgs.window-layout";
const QString NiftyMIDAS::PROP_BIND_WINDOWS = "applicationArgs.bind-windows";
const QString NiftyMIDAS::PROP_BIND_VIEWERS = "applicationArgs.bind-viewers";
const QString NiftyMIDAS::PROP_ANNOTATION = "applicationArgs.annotation";

}

/// \file NiftyMIDAS.cxx
/// \brief Main entry point for NiftyMIDAS application.
int main(int argc, char** argv)
{
  niftk::NiftyMIDAS app(argc, argv);

  return app.run();
}
