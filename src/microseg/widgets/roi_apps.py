'''
Base classes for building apps from ROI editors
'''
from .base import *
from .pg import *
from .roi_image import *

class ImageSegmentorApp(SaveableApp):
    '''
    Simple usable app for segmenting single (or stacks) of images in ZXYC format
    '''
    def __init__(self, img_path: str, desc: str='rois', *args, **kwargs):
        # State
        self._z = 0
        self._img_path = img_path
        self._img = load_stack(img_path) # ZXYC
        self._zmax = self._img.shape[0]
        self._rois = [[] for _ in range(self._zmax)]
        
        # Widgets
        self._main = VLayoutWidget()
        self._creator = ROIsCreator()
        self._main._layout.addWidget(self._creator)
        self._z_slider = IntegerSlider(mode='scroll')
        self._main._layout.addWidget(self._z_slider)
        if self._zmax == 1:
            print(f'Received standard 2D image, disabling z-slider')
            self._z_slider.hide()
        else:
            print(f'Received z-stack with {self._zmax} slices, enabling z-slider')
            self._z_slider.setData(0, self._zmax-1, self._z)
        self.setCentralWidget(self._main)

        # Listeners
        self._creator.proposeAdd.connect(self._add)
        self._creator.proposeDelete.connect(self._delete)
        self._z_slider.valueChanged.connect(lambda z: self._set_z(z))

        # Run data load and rest of initialization in superclass
        super().__init__(
            f'Image {desc} segmentor',
            f'{os.path.splitext(img_path)[0]}.{desc}',
        *args, **kwargs)

    ''' Overrides '''

    def copyIntoState(self, state: List[List[ROI]]):
        self._rois = [[r.copy() for r in subrois] for subrois in state]
        assert len(self._rois) == self._zmax, f'Expected {self._zmax} z-slices, got {len(self._rois)}'
        self._refresh_ROIs()

    def copyFromState(self) -> List[List[ROI]]:
        return [[r.copy() for r in subrois] for subrois in self._rois]

    def readData(path: str) -> List[List[ROI]]:
        return pickle.load(open(path, 'rb'))
    
    def writeData(path: str, data: List[List[ROI]]):
        pickle.dump(data, open(path, 'wb'))

    ''' Private methods '''

    def _refresh_ROIs(self):
        self._creator.setROIs(self._rois[self._z])

    def _set_z(self, z: int):
        assert 0 <= z <= self._zmax-1, f'Invalid z: {z}'
        self._z = z
        self._creator.setData(self._img[z], self._rois[z])

    @property
    def next_label(self) -> int:
        return max(
            [max([r.lbl for r in subrois], default=0) 
             for subrois in self._rois], 
            default=0
        )

    def _add(self, rois: List[ROI]):
        l = self.next_label
        lrois = [LabeledROI(l, r) for r in rois]
        self._rois[self._z].extend(lrois)
        self._refresh_ROIs()
        self.pushEdit()

    def _delete(self, rois: Set[int]):
        self._rois[self._z] = [
            r for r in self._rois[self._z] if not (r.lbl in rois)
        ]
        self._refresh_ROIs()
        self.pushEdit()