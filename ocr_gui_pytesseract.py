
import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pytesseract
from pytesseract import Output

# If tesseract isn't in PATH, uncomment and set path below (example Windows):
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ImageLabel(QtWidgets.QLabel):
    """QLabel that supports mouse-driven ROI selection and displays OpenCV images."""

    roi_changed = QtCore.pyqtSignal(tuple)  # x, y, w, h

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.start_pos = None
        self.end_pos = None
        self.selecting = False
        self._pixmap = None
        self.display_image = None  # BGR image currently shown

    def setImage(self, image_bgr):
        """Sets current image (OpenCV BGR) and shows it scaled to label."""
        self.display_image = image_bgr.copy() if image_bgr is not None else None
        if image_bgr is None:
            self.clear()
            return
        h, w, ch = image_bgr.shape
        bytes_per_line = ch * w
        # convert BGR -> RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(self._pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selecting and self._pixmap is not None and self.start_pos and self.end_pos:
            painter = QtGui.QPainter(self)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2, QtCore.Qt.DashLine))
            # draw rectangle in widget coordinates
            r = QtCore.QRect(self.start_pos, self.end_pos)
            painter.drawRect(r.normalized())

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.selecting = True
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.selecting:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.selecting:
            self.selecting = False
            self.end_pos = event.pos()
            self.update()
            # translate widget coords to image coords
            if self.display_image is None or self._pixmap is None:
                return
            pixmap_size = self.pixmap().size()
            label_size = self.size()
            # compute scale and top-left offset used in scaled pixmap
            scaled = self._pixmap.scaled(label_size, QtCore.Qt.KeepAspectRatio)
            sx = (label_size.width() - scaled.width()) // 2
            sy = (label_size.height() - scaled.height()) // 2
            # positions relative to scaled pixmap
            x1 = self.start_pos.x() - sx
            y1 = self.start_pos.y() - sy
            x2 = self.end_pos.x() - sx
            y2 = self.end_pos.y() - sy
            # clamp
            x1 = max(0, min(scaled.width(), x1))
            x2 = max(0, min(scaled.width(), x2))
            y1 = max(0, min(scaled.height(), y1))
            y2 = max(0, min(scaled.height(), y2))
            if x1 == x2 or y1 == y2:
                return
            # map to original image coordinates
            img_h, img_w = self.display_image.shape[:2]
            scale_x = img_w / scaled.width()
            scale_y = img_h / scaled.height()
            ix1 = int(min(x1, x2) * scale_x)
            iy1 = int(min(y1, y2) * scale_y)
            ix2 = int(max(x1, x2) * scale_x)
            iy2 = int(max(y1, y2) * scale_y)
            w = ix2 - ix1
            h = iy2 - iy1
            self.roi_changed.emit((ix1, iy1, w, h))


class OCRApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Printed Text Scanner - PyTesseract')
        self.resize(1100, 700)

        # UI Components
        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.roi_changed.connect(self.on_roi_changed)

        self.load_btn = QtWidgets.QPushButton('Load Image')
        self.load_btn.clicked.connect(self.load_image)

        self.start_cam_btn = QtWidgets.QPushButton('Start Camera')
        self.start_cam_btn.clicked.connect(self.toggle_camera)
        self.cam_running = False
        self.capture_btn = QtWidgets.QPushButton('Capture Frame')
        self.capture_btn.clicked.connect(self.capture_frame)

        self.ocr_btn = QtWidgets.QPushButton('Run OCR')
        self.ocr_btn.clicked.connect(self.run_ocr)

        self.clear_roi_btn = QtWidgets.QPushButton('Clear ROI')
        self.clear_roi_btn.clicked.connect(self.clear_roi)

        self.save_text_btn = QtWidgets.QPushButton('Save Text')
        self.save_text_btn.clicked.connect(self.save_text)

        # Preprocessing options
        self.cb_grayscale = QtWidgets.QCheckBox('Grayscale')
        self.cb_thresh = QtWidgets.QCheckBox('Threshold')
        self.cb_denoise = QtWidgets.QCheckBox('Denoise')
        self.cb_detect_boxes = QtWidgets.QCheckBox('Show Boxes')
        self.cb_detect_boxes.setChecked(True)

        # Text display
        self.text_area = QtWidgets.QPlainTextEdit()
        self.text_area.setReadOnly(False)

        # Layouts
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.image_label)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.load_btn)
        controls.addWidget(self.start_cam_btn)
        controls.addWidget(self.capture_btn)
        controls.addWidget(self.ocr_btn)
        controls.addWidget(self.clear_roi_btn)
        controls.addWidget(self.save_text_btn)
        left_layout.addLayout(controls)

        options_layout = QtWidgets.QHBoxLayout()
        options_layout.addWidget(self.cb_grayscale)
        options_layout.addWidget(self.cb_thresh)
        options_layout.addWidget(self.cb_denoise)
        options_layout.addWidget(self.cb_detect_boxes)
        left_layout.addLayout(options_layout)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addLayout(left_layout, 70)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(QtWidgets.QLabel('Extracted Text:'))
        right_layout.addWidget(self.text_area)
        main_layout.addLayout(right_layout, 30)

        # Internal state
        self.current_image = None  # BGR
        self.roi = None  # (x,y,w,h) or None

        # Camera
        self.cap = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.query_frame)

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Failed to load image')
            return
        self.current_image = img
        self.roi = None
        self.update_display()

    def toggle_camera(self):
        if not self.cam_running:
            # start
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(self, 'Error', 'Cannot open camera')
                return
            self.cam_running = True
            self.start_cam_btn.setText('Stop Camera')
            self.timer.start(30)
        else:
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.cam_running = False
            self.start_cam_btn.setText('Start Camera')

    def query_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        # store but don't overwrite current_image until capture
        self.live_frame = frame.copy()
        display = frame.copy()
        # if ROI present, draw it
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)
        self.image_label.setImage(display)

    def capture_frame(self):
        if hasattr(self, 'live_frame') and self.live_frame is not None:
            self.current_image = self.live_frame.copy()
            self.update_display()
        else:
            QtWidgets.QMessageBox.information(self, 'Info', 'No live frame to capture')

    def update_display(self, overlay_boxes=None):
        if self.current_image is None:
            self.image_label.setImage(None)
            return
        display = self.current_image.copy()
        # draw roi
        if self.roi is not None:
            x,y,w,h = self.roi
            cv2.rectangle(display, (x,y), (x+w, y+h), (0,255,0), 2)
        # overlay boxes (list of x,y,w,h,text)
        if overlay_boxes:
            for (bx,by,bw,bh,text) in overlay_boxes:
                cv2.rectangle(display, (bx,by), (bx+bw, by+bh), (0,0,255), 2)
                # draw text background
                cv2.rectangle(display, (bx, by-18), (bx+bw, by), (0,0,255), -1)
                cv2.putText(display, text[:30], (bx+2, by-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        self.image_label.setImage(display)

    def on_roi_changed(self, roi_tuple):
        self.roi = roi_tuple
        self.update_display()

    def clear_roi(self):
        self.roi = None
        self.update_display()

    def preprocess_for_ocr(self, img):
        # img: BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.cb_denoise.isChecked():
            gray = cv2.medianBlur(gray, 3)
        if self.cb_thresh.isChecked():
            # adaptive threshold
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        return gray

    def run_ocr(self):
        if self.current_image is None:
            QtWidgets.QMessageBox.information(self, 'Info', 'No image loaded or captured')
            return
        # select area
        img = self.current_image
        if self.roi is not None:
            x,y,w,h = self.roi
            img = img[y:y+h, x:x+w]
            if img.size == 0:
                QtWidgets.QMessageBox.warning(self, 'Error', 'ROI is empty')
                return
        # preprocess
        proc = self.preprocess_for_ocr(img)
        # pytesseract works best on PIL images or numpy; we will pass the numpy
        config = '--oem 3 --psm 3'  # default
        try:
            # get full text
            text = pytesseract.image_to_string(proc, config=config)
            self.text_area.setPlainText(text)
            overlay_boxes = []
            if self.cb_detect_boxes.isChecked():
                # get detailed data for bounding boxes
                d = pytesseract.image_to_data(proc, output_type=Output.DICT, config=config)
                n = len(d['level'])
                for i in range(n):
                    conf = int(d['conf'][i]) if d['conf'][i].isdigit() else -1
                    if conf > 30:
                        (bx, by, bw, bh) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                        word = d['text'][i]
                        # if ROI applied, need to adjust overlay coordinates to original image
                        if self.roi is not None:
                            rx, ry, _, _ = self.roi
                            bx_full = bx + rx
                            by_full = by + ry
                        else:
                            bx_full = bx
                            by_full = by
                        overlay_boxes.append((bx_full, by_full, bw, bh, word))
            else:
                overlay_boxes = None
            self.update_display(overlay_boxes)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Error', f'OCR failed: {e}')

    def save_text(self):
        txt = self.text_area.toPlainText()
        if not txt:
            QtWidgets.QMessageBox.information(self, 'Info', 'No text to save')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save text', 'extracted.txt', 'Text Files (*.txt)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(txt)
            QtWidgets.QMessageBox.information(self, 'Saved', f'Saved to {path}')
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Error', f'Failed to save: {e}')


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = OCRApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
