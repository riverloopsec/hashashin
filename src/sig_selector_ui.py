#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import re
import codecs
from PySide2.QtWidgets import (QLineEdit, QPushButton, QApplication, QTextEdit, QWidget,
    QVBoxLayout, QHBoxLayout, QDialog, QFileSystemModel, QTreeView, QLabel, QSplitter,
    QInputDialog, QMessageBox, QHeaderView, QMenu, QAction, QKeySequenceEdit,
    QPlainTextEdit)
from PySide2.QtCore import (QDir, QObject, Qt, QFileInfo, QItemSelectionModel, QSettings, QUrl)
from PySide2.QtGui import (QFont, QFontMetrics, QDesktopServices, QKeySequence, QIcon)
from binaryninja import user_plugin_path
from binaryninja.plugin import PluginCommand, MainThreadActionHandler
from binaryninja.mainthread import execute_on_main_thread
from binaryninja.log import (log_error, log_debug)
from binaryninjaui import (getMonospaceFont, UIAction, UIActionHandler, Menu, DockHandler)
import numbers

signatures_path = os.path.realpath(os.path.join(user_plugin_path(), "..", "sigs"))
try:
    if not os.path.exists(signatures_path):
        os.mkdir(signatures_path)
except IOError:
    log_error("Unable to create %s" % signatures_path)


def includeWalk(dir, includeExt):
    filePaths = []
    for (root, dirs, files) in os.walk(dir):
        for f in files:
            if os.path.splitext(f)[1] in includeExt:
                filePaths.append(os.path.join(root, f))
    return filePaths


def loadSnippetFromFile(snippetPath):
    try:
        snippetText = codecs.open(snippetPath, 'r', "utf-8").readlines()
    except:
        return ("", "", "")
    if (len(snippetText) < 3):
        return ("", "", "")
    else:
        qKeySequence = QKeySequence(snippetText[1].strip()[1:])
        if qKeySequence.isEmpty():
            qKeySequence = None
        return (snippetText[0].strip()[1:],
                qKeySequence,
                ''.join(snippetText[2:])
        )


class Signatures_UI(QDialog):

    def __init__(self, context, parent=None):
        super(Signatures_UI, self).__init__(parent)
        # Create widgets
        self.setWindowModality(Qt.ApplicationModal)
        self.title = QLabel(self.tr("Signatures"))
        self.saveButton = QPushButton(self.tr("&Save"))
        self.saveButton.setShortcut(QKeySequence(self.tr("Ctrl+S")))
        self.setWindowTitle(self.title.text())
        #self.newFolderButton = QPushButton("New Folder")
        self.browseButton = QPushButton("Browse Signatures")
        self.deleteSnippetButton = QPushButton("Delete")
        self.newSnippetButton = QPushButton("New Snippet")
        self.resetting = False
        self.columns = 3
        self.context = context

        self.keySequenceEdit = QKeySequenceEdit(self)
        self.currentHotkey = QKeySequence()
        self.currentHotkeyLabel = QLabel("")
        self.currentFileLabel = QLabel()
        self.currentFile = ""

        font = getMonospaceFont(self)

        #Files
        self.files = QFileSystemModel()
        self.files.setRootPath(signatures_path)
        self.files.setNameFilters(["*.py"])

        #Tree
        self.tree = QTreeView()
        self.tree.setModel(self.files)
        self.tree.setSortingEnabled(True)
        self.tree.hideColumn(2)
        self.tree.sortByColumn(0, Qt.AscendingOrder)
        self.tree.setRootIndex(self.files.index(signatures_path))
        for x in range(self.columns):
            #self.tree.resizeColumnToContents(x)
            self.tree.header().setSectionResizeMode(x, QHeaderView.ResizeToContents)
        treeLayout = QVBoxLayout()
        treeLayout.addWidget(self.tree)
        treeButtons = QHBoxLayout()
        #treeButtons.addWidget(self.newFolderButton)
        treeButtons.addWidget(self.browseButton)
        treeButtons.addWidget(self.newSnippetButton)
        treeButtons.addWidget(self.deleteSnippetButton)
        treeLayout.addLayout(treeButtons)
        treeWidget = QWidget()
        treeWidget.setLayout(treeLayout)

        # Create layout and add widgets
        buttons = QHBoxLayout()
        buttons.addWidget(self.clearHotkeyButton)
        buttons.addWidget(self.keySequenceEdit)
        buttons.addWidget(self.currentHotkeyLabel)
        buttons.addWidget(self.saveButton)

        description = QHBoxLayout()
        description.addWidget(QLabel(self.tr("Description: ")))
        description.addWidget(self.snippetDescription)

        vlayoutWidget = QWidget()
        vlayout = QVBoxLayout()
        vlayout.addLayout(description)
        vlayout.addWidget(self.edit)
        vlayout.addLayout(buttons)
        vlayoutWidget.setLayout(vlayout)

        hsplitter = QSplitter()
        hsplitter.addWidget(treeWidget)
        hsplitter.addWidget(vlayoutWidget)

        hlayout = QHBoxLayout()
        hlayout.addWidget(hsplitter)

        self.showNormal() #Fixes bug that maximized windows are "stuck"
        #Because you can't trust QT to do the right thing here
        if (sys.platform == "darwin"):
            self.settings = QSettings("Vector35", "Snippet Editor")
        else:
            self.settings = QSettings("Vector 35", "Snippet Editor")
        if self.settings.contains("ui/snippeteditor/geometry"):
            self.restoreGeometry(self.settings.value("ui/snippeteditor/geometry"))
        else:
            self.edit.setMinimumWidth(80 * font.averageCharWidth())
            self.edit.setMinimumHeight(30 * font.lineSpacing())

        # Set dialog layout
        self.setLayout(hlayout)

        # Add signals
        self.saveButton.clicked.connect(self.save)
        self.clearHotkeyButton.clicked.connect(self.clearHotkey)
        self.tree.selectionModel().selectionChanged.connect(self.selectFile)
        self.newSnippetButton.clicked.connect(self.newFileDialog)
        self.deleteSnippetButton.clicked.connect(self.deleteSnippet)
        #self.newFolderButton.clicked.connect(self.newFolder)
        self.browseButton.clicked.connect(self.browseSnippets)

        if self.settings.contains("ui/snippeteditor/selected"):
            selectedName = self.settings.value("ui/snippeteditor/selected")
            self.tree.selectionModel().select(self.files.index(selectedName), QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
            if self.tree.selectionModel().hasSelection():
                self.selectFile(self.tree.selectionModel().selection(), None)
                self.edit.setFocus()
                cursor = self.edit.textCursor()
                cursor.setPosition(self.edit.document().characterCount()-1)
                self.edit.setTextCursor(cursor)
            else:
                self.readOnly(True)
        else:
            self.readOnly(True)


    @staticmethod
    def registerAllSnippets():
        for action in list(filter(lambda x: x.startswith("Snippets\\"), UIAction.getAllRegisteredActions())):
            if action == "Snippets\\Snippet Editor...":
                continue
            UIActionHandler.globalActions().unbindAction(action)
            Menu.mainMenu("Tools").removeAction(action)
            UIAction.unregisterAction(action)

        for snippet in includeWalk(signatures_path, ".py"):
            print(snippet)
            '''          
            snippetKeys = None
            (snippetDescription, snippetKeys, snippetCode) = loadSnippetFromFile(snippet)
            actionText = actionFromSnippet(snippet, snippetDescription)
            if snippetCode:
                if snippetKeys == None:
                    UIAction.registerAction(actionText)
                else:
                    UIAction.registerAction(actionText, snippetKeys)
                UIActionHandler.globalActions().bindAction(actionText, UIAction(makeSnippetFunction(snippetCode)))
                Menu.mainMenu("Tools").addAction(actionText, "Snippets")
                '''


    def clearSelection(self):
        self.keySequenceEdit.clear()
        self.currentHotkey = QKeySequence()
        self.currentHotkeyLabel.setText("")
        self.currentFileLabel.setText("")
        self.snippetDescription.setText("")
        self.edit.setPlainText("")
        self.currentFile = ""

    def reject(self):
        self.settings.setValue("ui/snippeteditor/geometry", self.saveGeometry())

        if self.snippetChanged():
            question = QMessageBox.question(self, self.tr("Discard"), self.tr("You have unsaved changes, quit anyway?"))
            if question != QMessageBox.StandardButton.Yes:
                return
        self.accept()

    def browseSnippets(self):
        url = QUrl.fromLocalFile(signatures_path)
        QDesktopServices.openUrl(url);

    def newFolder(self):
        (folderName, ok) = QInputDialog.getText(self, self.tr("Folder Name"), self.tr("Folder Name: "))
        if ok and folderName:
            index = self.tree.selectionModel().currentIndex()
            selection = self.files.filePath(index)
            if QFileInfo(selection).isDir():
                QDir(selection).mkdir(folderName)
            else:
                QDir(signatures_path).mkdir(folderName)

    def selectFile(self, new, old):
        if (self.resetting):
            self.resetting = False
            return
        newSelection = self.files.filePath(new.indexes()[0])
        self.settings.setValue("ui/snippeteditor/selected", newSelection)
        if QFileInfo(newSelection).isDir():
            self.readOnly(True)
            self.tree.clearSelection()
            self.currentFile = ""
            return

        if old and old.length() > 0:
            oldSelection = self.files.filePath(old.indexes()[0])
            if not QFileInfo(oldSelection).isDir() and self.snippetChanged():
                question = QMessageBox.question(self, self.tr("Discard"), self.tr("Snippet changed. Discard changes?"))
                if question != QMessageBox.StandardButton.Yes:
                    self.resetting = True
                    self.tree.selectionModel().select(old, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                    return False

        self.currentFile = newSelection
        self.loadSnippet()

    def loadSnippet(self):
        self.currentFileLabel.setText(QFileInfo(self.currentFile).baseName())
        (snippetDescription, snippetKeys, snippetCode) = loadSnippetFromFile(self.currentFile)
        self.snippetDescription.setText(snippetDescription) if snippetDescription else self.snippetDescription.setText("")
        self.keySequenceEdit.setKeySequence(snippetKeys) if snippetKeys else self.keySequenceEdit.setKeySequence(QKeySequence(""))
        self.edit.setPlainText(snippetCode) if snippetCode else self.edit.setPlainText("")
        self.readOnly(False)

    def newFileDialog(self):
        (snippetName, ok) = QInputDialog.getText(self, self.tr("Snippet Name"), self.tr("Snippet Name: "))
        if ok and snippetName:
            if not snippetName.endswith(".py"):
                snippetName += ".py"
            index = self.tree.selectionModel().currentIndex()
            selection = self.files.filePath(index)
            if QFileInfo(selection).isDir():
                path = os.path.join(selection, snippetName)
            else:
                path = os.path.join(signatures_path, snippetName)
                self.readOnly(False)
            open(path, "w").close()
            self.tree.setCurrentIndex(self.files.index(path))
            log_debug("Snippet %s created." % snippetName)

    def readOnly(self, flag):
        self.keySequenceEdit.setEnabled(not flag)
        self.snippetDescription.setReadOnly(flag)
        self.edit.setReadOnly(flag)
        if flag:
            self.snippetDescription.setDisabled(True)
            self.edit.setDisabled(True)
        else:
            self.snippetDescription.setEnabled(True)
            self.edit.setEnabled(True)

def launchPlugin(context):
    snippets = Signatures_UI(context)
    snippets.exec_()
