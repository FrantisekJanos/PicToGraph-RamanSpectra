import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLineEdit, QSizePolicy, QMessageBox, QStatusBar
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QIcon, QImage
from PyQt5.QtCore import Qt, QRect, QPoint

from simple_line import preprocess_image_from_array, extract_and_plot_contour
from find_peaks import plot_spectrum_with_peaks

import matplotlib.pyplot as plt
import io
import numpy as np
import csv


def qpixmap_to_array(pixmap):
    # Převod QPixmap na QImage
    qimage = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    # Vrátíme pouze RGB kanály
    return arr[..., :3]

class MagnifierLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip)
        self.setFixedSize(100, 100)
        self.setStyleSheet("border: 2px solid black;")
        self.magnified_pixmap = None  # Uchováme zvětšený obrázek

    def setMagnifiedPixmap(self, pixmap):
        """ Nastaví zvětšený obrázek a překreslí widget """
        self.magnified_pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        """ Překreslí lupu a přidá zaměřovací kříž """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)  # Povolíme antialiasing
        if self.magnified_pixmap:
            painter.drawPixmap(0, 0, self.magnified_pixmap)

        # Přidání zaměřovacího kříže
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        center_x = self.width() // 2
        center_y = self.height() // 2

        # Vykreslení horizontální a vertikální čáry
        painter.drawLine(center_x, 0, center_x, self.height())  # Svislá čára
        painter.drawLine(0, center_y, self.width(), center_y)  # Vodorovná čára

class CropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.selection_rect = None
        self.drawing = False  # Indikuje, zda probíhá kreslení výběru
        self.setMouseTracking(True)
        self.magnifier = MagnifierLabel()
        # Atributy pro režim kříže (pravítko)
        self.show_crosshair = False
        self.current_cursor_pos = None
    def enterEvent(self, event):
        # Zobrazí lupu, jakmile kurzor vstoupí do widgetu
        self.magnifier.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Skryje lupu, když kurzor opustí widget
        self.magnifier.hide()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.selection_rect = QRect(self.start_point, self.end_point)
            self.drawing = True
            self.updateMagnifier(event)
            self.update()

    # def mouseMoveEvent(self, event):
    #     self.current_cursor_pos = event.pos()
    #     if self.drawing:
    #         self.end_point = event.pos()
    #         self.selection_rect = QRect(self.start_point, self.end_point).normalized()
    #     self.updateMagnifier(event)
    #     if self.show_crosshair:
    #         self.update()
    def mouseMoveEvent(self, event):
        self.current_cursor_pos = event.pos()
        # Vždy aktualizujeme lupu, aby byla viditelná i před kliknutím
        self.updateMagnifier(event)
        if self.drawing:
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
        self.update()  # Překreslí widget, aby byl výběrový rámeček viditelný

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.drawing = False
            self.update()

    # def updateMagnifier(self, event):
    #     base_pixmap = self.pixmap()
    #     if base_pixmap is None:
    #         self.magnifier.hide()
    #         return
    #     label_width = self.width()
    #     label_height = self.height()
    #     displayed_width = base_pixmap.width()
    #     displayed_height = base_pixmap.height()
    #     offset_x = (label_width - displayed_width) // 2
    #     offset_y = (label_height - displayed_height) // 2
    #
    #     pixmap_pos = event.pos() - QPoint(offset_x, offset_y)
    #     if (pixmap_pos.x() < 0 or pixmap_pos.y() < 0 or
    #             pixmap_pos.x() >= displayed_width or pixmap_pos.y() >= displayed_height):
    #         self.magnifier.show()
    #         return
    #
    #     region_size = 30
    #     half = region_size // 2
    #
    #
    #     x = pixmap_pos.x() - half
    #     y = pixmap_pos.y() - half
    #     if x < 0:
    #         x = 0
    #     if y < 0:
    #         y = 0
    #     if x + region_size > displayed_width:
    #         x = displayed_width - region_size
    #     if y + region_size > displayed_height:
    #         y = displayed_height - region_size
    #
    #     region = base_pixmap.copy(x, y, region_size, region_size)
    #     magnified = region.scaled(self.magnifier.width(), self.magnifier.height(), Qt.KeepAspectRatio,
    #                               Qt.SmoothTransformation)
    #     # self.magnifier.setPixmap(magnified)
    #
    #     # Použití nové metody k aktualizaci zvětšené pixmapy
    #     self.magnifier.setMagnifiedPixmap(magnified)
    #
    #     global_pos = self.mapToGlobal(event.pos())
    #     offset = 20
    #     self.magnifier.move(global_pos.x() + offset, global_pos.y() + offset)

    def updateMagnifier(self, event):
        base_pixmap = self.pixmap()
        if base_pixmap is None:
            self.magnifier.hide()
            return

        label_width = self.width()
        label_height = self.height()
        displayed_width = base_pixmap.width()
        displayed_height = base_pixmap.height()
        offset_x = (label_width - displayed_width) // 2
        offset_y = (label_height - displayed_height) // 2

        # Získáme pozici kurzoru relativně k pixmapě
        pixmap_pos = event.pos() - QPoint(offset_x, offset_y)

        region_size = 30
        half = region_size // 2

        # Vytvoříme novou pixmapu s průhledným pozadím pro danou oblast
        region_pixmap = QPixmap(region_size, region_size)
        region_pixmap.fill(Qt.transparent)
        painter = QPainter(region_pixmap)

        # Definujeme zdrojovou oblast, jejíž střed má odpovídat poloze kurzoru
        src_x = pixmap_pos.x() - half
        src_y = pixmap_pos.y() - half
        src_rect = QRect(src_x, src_y, region_size, region_size)

        # Výchozí cílová oblast je celá oblast region_pixmap
        dest_rect = QRect(0, 0, region_size, region_size)

        # Pokud je zdrojová oblast částečně mimo hranice obrázku, upravíme ji a zároveň odpovídající část cílové oblasti
        if src_rect.left() < 0:
            diff = -src_rect.left()
            src_rect.setLeft(0)
            dest_rect.setLeft(diff)
        if src_rect.top() < 0:
            diff = -src_rect.top()
            src_rect.setTop(0)
            dest_rect.setTop(diff)
        if src_rect.right() > displayed_width:
            diff = src_rect.right() - displayed_width
            src_rect.setRight(displayed_width)
            dest_rect.setRight(region_size - diff)
        if src_rect.bottom() > displayed_height:
            diff = src_rect.bottom() - displayed_height
            src_rect.setBottom(displayed_height)
            dest_rect.setBottom(region_size - diff)

        # Vykreslíme část obrázku z base_pixmap do naší region_pixmap
        painter.drawPixmap(dest_rect, base_pixmap, src_rect)
        painter.end()

        # Zvětšíme region na velikost lupy
        magnified = region_pixmap.scaled(self.magnifier.width(), self.magnifier.height(),
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.magnifier.setMagnifiedPixmap(magnified)
        global_pos = self.mapToGlobal(event.pos())
        offset = 20
        self.magnifier.move(global_pos.x() + offset, global_pos.y() + offset)
        self.magnifier.show()
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)  # Povolíme antialiasing pro hladké vykreslení
        if self.selection_rect:
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
        if self.show_crosshair and self.current_cursor_pos:
            cross_pen = QPen(Qt.green, 1, Qt.DashLine)
            painter.setPen(cross_pen)
            painter.drawLine(0, self.current_cursor_pos.y(), self.width(), self.current_cursor_pos.y())
            painter.drawLine(self.current_cursor_pos.x(), 0, self.current_cursor_pos.x(), self.height())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PicToGraphApp - Raman Base")
        self.original_pixmap = None
        self.last_x = None  # Inicializace spektra
        self.last_y = None
        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Přidání status baru pro zpětnou vazbu
        self.setStatusBar(QStatusBar())

        # Horní část: obrázky
        image_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)

        self.label_original = CropLabel()
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("background-color: #ccc;")
        image_layout.addWidget(self.label_original)

        # Pravá část – rozdělená do dvou vertikálních oblastí
        right_layout = QVBoxLayout()
        self.label_cropped = QLabel("Oříznutý obrázek")
        self.label_cropped.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.label_cropped)

        self.label_result = QLabel("Výsledek funkce se zobrazí zde")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("background-color: #eee; border: 1px solid #ccc;")
        right_layout.addWidget(self.label_result)

        image_layout.addLayout(right_layout)

        # Řádek se vstupními poli na jediné řádce – minimální okraje a mezery
        param_layout = QHBoxLayout()
        param_layout.setContentsMargins(0, 0, 0, 0)  # minimální okraje
        param_layout.setSpacing(5)  # malá mezera mezi widgety

        # Xmin
        label_xmin = QLabel("Xmin:")
        label_xmin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(label_xmin)

        self.input_xmin = QLineEdit()
        self.input_xmin.setFixedWidth(50)
        self.input_xmin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(self.input_xmin)

        # Xmax
        label_xmax = QLabel("Xmax:")
        label_xmax.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(label_xmax)

        self.input_xmax = QLineEdit()
        self.input_xmax.setFixedWidth(50)
        self.input_xmax.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(self.input_xmax)

        # Ymin
        label_ymin = QLabel("Ymin:")
        label_ymin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(label_ymin)

        self.input_ymin = QLineEdit()
        self.input_ymin.setFixedWidth(50)
        self.input_ymin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(self.input_ymin)

        # Ymax
        label_ymax = QLabel("Ymax:")
        label_ymax.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(label_ymax)

        self.input_ymax = QLineEdit()
        self.input_ymax.setFixedWidth(50)
        self.input_ymax.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(self.input_ymax)

        # Find peaks
        sensitivity_label = QLabel("Sensitivity:")
        self.input_sensitivity = QLineEdit()
        self.input_sensitivity.setFixedWidth(50)
        min_distance_label = QLabel("Min distance:")
        self.input_min_distance = QLineEdit()
        self.input_min_distance.setFixedWidth(50)
        param_layout.addWidget(sensitivity_label)
        param_layout.addWidget(self.input_sensitivity)
        param_layout.addWidget(min_distance_label)
        param_layout.addWidget(self.input_min_distance)


        # Přidá stretch na konec, aby zbytek řádku zůstal prázdný
        param_layout.addStretch(1)

        main_layout.addLayout(param_layout)

        # Spodní část: tlačítka
        btn_load = QPushButton("Načíst obrázek")
        btn_load.clicked.connect(self.load_image)
        main_layout.addWidget(btn_load)

        btn_crop = QPushButton("Oříznout obrázek")
        btn_crop.clicked.connect(self.crop_image)
        main_layout.addWidget(btn_crop)

        btn_process = QPushButton("Zpracovat spektrum")
        btn_process.clicked.connect(self.process_cropped_image)
        main_layout.addWidget(btn_process)

        btn_find_peaks = QPushButton("Find Peaks")
        btn_find_peaks.clicked.connect(self.find_peaks)
        main_layout.addWidget(btn_find_peaks)

        # Nové tlačítko pro export do CSV
        btn_export = QPushButton("Export to CSV")
        btn_export.clicked.connect(self.export_to_csv)
        main_layout.addWidget(btn_export)

        btn_crosshair = QPushButton("Zaměřování")
        btn_crosshair.setCheckable(True)
        btn_crosshair.clicked.connect(self.toggle_crosshair)
        main_layout.addWidget(btn_crosshair)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Otevřít obrázek", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.original_pixmap = QPixmap(file_name)
            self.display_image = self.original_pixmap.scaled(
                self.label_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.label_original.setPixmap(self.display_image)
            self.label_cropped.setText("Oříznutý obrázek")
            self.label_result.setText("Výsledek funkce se zobrazí zde")
            self.label_original.selection_rect = None
            self.statusBar().showMessage("Obrázek načten.", 3000)
        else:
            QMessageBox.warning(self, "Chyba", "Nebyl vybrán žádný obrázek.")

    def crop_image(self):
        if self.original_pixmap and self.label_original.selection_rect:
            displayed = self.label_original.pixmap()
            if not displayed:
                return

            displayed_width = displayed.width()
            displayed_height = displayed.height()
            label_width = self.label_original.width()
            label_height = self.label_original.height()
            offset_x = (label_width - displayed_width) // 2
            offset_y = (label_height - displayed_height) // 2

            sel = self.label_original.selection_rect
            x = max(sel.x() - offset_x, 0)
            y = max(sel.y() - offset_y, 0)
            w = sel.width()
            h = sel.height()
            if x + w > displayed_width:
                w = displayed_width - x
            if y + h > displayed_height:
                h = displayed_height - y

            scale_x = self.original_pixmap.width() / displayed_width
            scale_y = self.original_pixmap.height() / displayed_height

            orig_rect = QRect(
                int(x * scale_x),
                int(y * scale_y),
                int(w * scale_x),
                int(h * scale_y)
            )
            cropped_pixmap = self.original_pixmap.copy(orig_rect)

            # Omezíme výšku oříznutého obrázku na max 300 pixelů
            max_cropped_height = 250  # Můžeš upravit dle potřeby
            scaled_cropped = cropped_pixmap.scaledToHeight(max_cropped_height, Qt.SmoothTransformation)

            self.label_cropped.setPixmap(scaled_cropped)

            # self.label_cropped.setPixmap(
            #     cropped_pixmap.scaled(self.label_cropped.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # )
        else:
            print("Obrázek nebyl načten nebo nebyla vybrána oblast!")

    def toggle_crosshair(self, checked):
        self.label_original.show_crosshair = checked
        self.label_original.update()

    def process_cropped_image(self):
        """
        1) Zkontroluje, zda máme v label_cropped oříznutý obrázek.
        2) Převede QPixmap na numpy pole pomocí qpixmap_to_array.
        3) Načte hodnoty x_min, x_max, y_min, y_max z QLineEdit.
        4) Zavolá funkci preprocess_image_from_array (importovanou ze simple_line.py),
           která vrátí NumPy pole obrázku a hlavní konturu.
        5) Vykreslí graf spektra pomocí extract_and_plot_contour (také ze simple_line.py)
           a uloží výsledek do paměti.
        6) Výsledný graf zobrazí v label_result.
        """
        cropped_pixmap = self.label_cropped.pixmap()
        if not cropped_pixmap:
            # self.label_result.setText("Chybí oříznutý obrázek!")
            QMessageBox.warning(self, "Chyba", "Chybí oříznutý obrázek!")
            return

        try:
            # Načtení hodnot z QLineEdit
            x_min = float(self.input_xmin.text())
            x_max = float(self.input_xmax.text())
            y_min = float(self.input_ymin.text())
            y_max = float(self.input_ymax.text())
        except ValueError:
            # self.label_result.setText("Chybné hodnoty Xmin/Xmax/Ymin/Ymax!")
            QMessageBox.warning(self, "Chyba", "Chybné hodnoty Xmin/Xmax/Ymin/Ymax!")
            return

        try:
            # Vypneme interaktivní režim matplotlib
            plt.ioff()

            # Převod QPixmap na numpy pole
            img_array = qpixmap_to_array(cropped_pixmap)

            # Použijeme funkci preprocess_image_from_array, která očekává numpy pole
            # Ujisti se, že jsi tuto funkci importoval, např.:
            # from simple_line import preprocess_image_from_array, extract_and_plot_contour
            img, main_contour = preprocess_image_from_array(img_array)

            # Vykreslíme graf do matplotlibu (bez plt.show())
            # extract_and_plot_contour(img, main_contour, x_min, x_max, y_min, y_max)

            # Vykreslíme graf a zároveň získáme data spektra (data_x, data_y)
            data_x, data_y = extract_and_plot_contour(img, main_contour, x_min, x_max, y_min, y_max)

            # Uložíme spektrum pro použití funkcí (např. v tlačítku "find peaks")
            self.last_x = data_x
            self.last_y = data_y

            # Uložíme vykreslený graf do paměti
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()  # Zavře figure
            buf.seek(0)

            # Vytvoříme QImage z bytestreamu a nastavíme jej do label_result
            qimage = QImage.fromData(buf.getvalue(), 'PNG')
            if qimage.isNull():
                QMessageBox.critical(self, "Chyba", "Nepodařilo se vykreslit graf.")
                # self.label_result.setText("Nepodařilo se vykreslit graf.")
            else:
                pixmap = QPixmap.fromImage(qimage)
                # Omezíme maximální výšku výsledného grafu, aby se nezvětšoval vertikálně
                max_height = 500  # nastav dle potřeby
                if pixmap.height() > max_height:
                    pixmap = pixmap.scaledToHeight(max_height, Qt.SmoothTransformation)
                self.label_result.setPixmap(pixmap)
                self.statusBar().showMessage("Spektrum bylo úspěšně zpracováno.", 3000)
        except Exception as e:
            # self.label_result.setText(f"Nastala chyba při zpracování: {e}")
            QMessageBox.critical(self, "Chyba", f"Nastala chyba při zpracování: {e}")
        finally:
            plt.ioff()

    def export_to_csv(self):
        """
        Exportuje spektrum uložené v self.last_x a self.last_y do CSV.
        """
        if self.last_x is None or self.last_y is None:
            # self.label_result.setText("Spektrum ještě nebylo vygenerováno!")
            QMessageBox.warning(self, "Chyba", "Spektrum ještě nebylo vygenerováno!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export to CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y"])
                for xi, yi in zip(self.last_x, self.last_y):
                    writer.writerow([xi, yi])
            # self.label_result.setText("Export byl úspěšný.")
            QMessageBox.information(self, "Úspěch", "Export byl úspěšný.")
            self.statusBar().showMessage("Export byl úspěšný.", 3000)
        except Exception as e:
            # self.label_result.setText(f"Export selhal: {e}")
            QMessageBox.critical(self, "Chyba", f"Export selhal: {e}")


    def find_peaks(self):
        try:
            sensitivity = float(self.input_sensitivity.text())
            min_distance = int(self.input_min_distance.text())
        except ValueError:
            # self.label_result.setText("Chybná hodnota pro sensitivity nebo min_distance!")
            QMessageBox.warning(self, "Chyba", "Chybná hodnota pro sensitivity nebo min_distance!")
            return

        # Předpokládáme, že spektrum (x a y) bylo dříve vygenerováno a uloženo do self.last_x, self.last_y
        if not hasattr(self, 'last_x') or not hasattr(self, 'last_y'):
            # self.label_result.setText("Spektrum ještě nebylo vygenerováno!")
            QMessageBox.warning(self, "Chyba", "Spektrum ještě nebylo vygenerováno!")
            return

        # Zavoláme funkci pro hledání peaků s parametry
        plot_spectrum_with_peaks(self.last_x, self.last_y, sensitivity, min_distance, show_peaks=True)
        self.statusBar().showMessage("Peak detection proběhla úspěšně.", 3000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
