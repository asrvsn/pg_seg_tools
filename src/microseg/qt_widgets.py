'''
General PyQT widgets
'''
from typing import List
import abc
from pyqtgraph.Qt import QtCore # ?
from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QFileSystemModel, QHeaderView, QLabel, QSizePolicy, QTableWidget, QTreeView, QVBoxLayout, QWidget, QGraphicsOpacityEffect

''' Layouts '''

class HLayout(QtWidgets.QHBoxLayout):
	'''
	HBoxLayout without margins
	'''
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setContentsMargins(0, 0, 0, 0)
		self.setSpacing(2)
		self.setAlignment(Qt.AlignmentFlag.AlignLeft)

class VLayout(QtWidgets.QVBoxLayout):
	'''
	VBoxLayout without margins
	'''
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setContentsMargins(0, 0, 0, 0)
		self.setSpacing(0)
		self.setAlignment(Qt.AlignmentFlag.AlignTop)

''' Widgets '''

class PushButton(QtWidgets.QPushButton):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setContentsMargins(0, 0, 0, 0)
		# self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

class MainWindow(QtWidgets.QMainWindow):

	def resizeToActiveScreen(self):
		screen = QApplication.primaryScreen()
		self.move(screen.geometry().center())
		self.resize(screen.size())

class TreeViewLeafSelectable(QTreeView):
	''' Version of QTreeView in which only leaf nodes are selectable '''
	def selectionCommand(self, index, event=None):
		if index.isValid() and not index.model().hasChildren(index):
			return QTreeView.selectionCommand(self, index, event)
		
class ExtensionViewer(QWidget):
	file_selected = QtCore.pyqtSignal(str)
	
	def __init__(self, folder_path, name_filters: List[str]=['*.*'], show_filtered: bool=False, parent=None):
		super(ExtensionViewer, self).__init__(parent)

		self.folder_path = folder_path

		layout = QVBoxLayout(self)

		model = QFileSystemModel()
		model.setRootPath(self.folder_path)
		model.setNameFilters(name_filters)
		model.setNameFilterDisables(show_filtered)

		tree_view = TreeViewLeafSelectable()
		tree_view.setModel(model)
		tree_view.setRootIndex(model.index(self.folder_path))
		tree_view.setSortingEnabled(True)
		
		# Show only name column
		for i in range(1, tree_view.model().columnCount()):
			tree_view.hideColumn(i)

		layout.addWidget(tree_view)
		self.setLayout(layout)

		tree_view.selectionModel().selectionChanged.connect(self.on_selection_changed)

	def on_selection_changed(self, selected):
		for ix in selected.indexes():
			if ix.column() == 0:
				path = self.sender().model().filePath(ix)
				self.file_selected.emit(path)

class StretchTableWidget(QTableWidget):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setMinimumSize(1, 1)
		self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
		self.setShowGrid(False)
		self.setAlternatingRowColors(True)
		# # Set vertical size to minimum
		# self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
		# Remove column and row labels
		self.verticalHeader().setVisible(False)
		self.horizontalHeader().setVisible(False)

class FlashOverlay(QWidget):
	def __init__(self, parent):
		super(FlashOverlay, self).__init__(parent)
		self._parent = parent
		self.setWindowFlags(Qt.WindowType.WindowTransparentForInput | Qt.WindowType.WindowStaysOnTopHint)
		self.opacity_effect = QGraphicsOpacityEffect(self)
		self.setGraphicsEffect(self.opacity_effect)
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.fade_out)

		# Add centered text
		self.text = QLabel(self, text='Saved')
		self.text.setAlignment(Qt.AlignCenter)

	def flash(self):
		self.setGeometry(self._parent.geometry())
		self.setWindowOpacity(1.0)
		self.timer.start(50)

	def fade_out(self):
		opacity = self.windowOpacity()
		if opacity < 0.0:
			self.setWindowOpacity(opacity - 0.1)
		else:
			self.setWindowOpacity(0)
			self.timer.stop()

class QtABCMeta(type(QtCore.QObject), abc.ABCMeta):
	pass