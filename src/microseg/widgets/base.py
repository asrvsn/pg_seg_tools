'''
General PyQT widgets
'''
from typing import List, Tuple, Union
import abc
from qtpy import QtCore
from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt, QTimer, QObject
from qtpy.QtWidgets import QApplication, QFileSystemModel, QHeaderView, QLabel, QSizePolicy, QTableWidget, QTreeView, QVBoxLayout, QWidget, QGraphicsOpacityEffect, QSlider, QScrollBar
from superqt import QRangeSlider

from .layout import *

''' Metaclasses '''

class QtABCMeta(type(QtCore.QObject), abc.ABCMeta):
	pass

class ClickProxy(QObject):
    sigClicked = QtCore.pyqtSignal()

''' Parent classes '''

class PaneledWidget(QWidget):
    '''
    Widget with bottom panel
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._layout = VLayout()
        self.setLayout(self._layout)
        self._main_layout = VLayout()
        self._main_widget = QWidget()
        self._main_widget.setLayout(self._main_layout)
        self._layout.addWidget(self._main_widget)
        self._bottom_layout = HLayout()
        self._bottom_widget = QWidget()
        self._bottom_widget.setLayout(self._bottom_layout)
        self._layout.addWidget(self._bottom_widget)

class SaveableWidget(PaneledWidget, metaclass=QtABCMeta):
    '''
    Basic panel widget containing a top and bottom layout, with settings on the bottom and a save button with Ctrl+S shortcut
    '''
    saved = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._settings_layout = HLayout()
        self._settings_widget = QWidget()
        self._settings_widget.setLayout(self._settings_layout)
        self._bottom_layout.addWidget(self._settings_widget)
        self._save_btn = PushButton('Save')
        self._bottom_layout.addWidget(self._save_btn)
        # self._overlay = FlashOverlay(self)
        self.setDisabled(True) # Initially disabled

        # Listeners
        self._save_btn.clicked.connect(self._on_save)

    def _on_save(self):
        self.saved.emit()
        # self._overlay.flash()

    @abc.abstractmethod
    def getData(self):
        '''
        Get data on save
        '''
        pass

    def keyPressEvent(self, ev):
        # Check for Ctrl+Save event
        if ev.key() == QtCore.Qt.Key_S and ev.modifiers() & QtCore.Qt.ControlModifier:
            self._on_save()

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

class IntegerSlider(HLayoutWidget):
    '''
    Integer slider with single handle
    '''
    def __init__(self, *args, mode: str='slide', **kwargs):
        super().__init__(*args, **kwargs)
        if mode == 'slide':
            self._slider = QSlider()
            self._slider.setTickPosition(QSlider.TicksBelow)
            self._slider.setTickInterval(1)
        elif mode == 'scroll':
            self._slider = QScrollBar()
        self._layout.addWidget(self._slider)
        self._label = QLabel()
        self._layout.addWidget(self._label)
        self._slider.setOrientation(QtCore.Qt.Horizontal)
        self._slider.setSingleStep(1)
        self._slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._slider.valueChanged.connect(lambda x: self._label.setText(f'{x}/{self._max}'))
        self.valueChanged = self._slider.valueChanged

    def setData(self, min: int, max: int, x: int):
        assert min <= x <= max, 'x must be within min and max'
        self._max = max
        self._slider.setMinimum(min)
        self._slider.setMaximum(max)
        self._slider.setValue(x)
        self._label.setText(f'{x}/{max}')

class IntegerRangeSlider(HLayoutWidget):
    '''
    Integer slider with two handles
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slider = QRangeSlider()
        self._layout.addWidget(self._slider)
        self._label = QLabel()
        self._layout.addWidget(self._label)
        self._slider.setOrientation(QtCore.Qt.Horizontal)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setSingleStep(1)
        self._slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._slider.valueChanged.connect(lambda x,y: self._label.setText(f'{x}-{y}/{self._max}'))
        self.valueChanged = self._slider.valueChanged

    def setData(self, min: int, max: int, range: Tuple[int, int]):
        x, y = range
        assert min <= x <= y <= max, 'range must be within min and max'
        self._max = max
        self._slider.setMinimum(min)
        self._slider.setMaximum(max)
        self._slider.setValue(range)
        self._label.setText(f'{x}-{y}/{max}')
