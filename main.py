import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLineEdit, QSizePolicy, QMessageBox, QStatusBar, QDialog, QScrollArea, QColorDialog
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QIcon, QImage, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, QRect, QPoint

from simple_line import preprocess_image_from_array, extract_and_plot_contour
from find_peaks import plot_spectrum_with_peaks
from clustering import preprocess_image, display_clusters, check_clusters_embedded
from functools import partial

import matplotlib.pyplot as plt
import io
import numpy as np
import csv
import tempfile

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


class ClusterWindow(QWidget):
    def __init__(self, cropped_pixmap=None, target_label=None):
        super().__init__()
        self.setWindowTitle("Cluster Window")
        self.cropped_pixmap = cropped_pixmap
        self.target_label = target_label  # Uložíme referenci na cílový widget
        self.init_ui(cropped_pixmap)
        self.showMaximized()

    def init_ui(self, cropped_pixmap):
        self.layout = QVBoxLayout(self)

        # Zobrazení oříznutého obrázku
        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        if cropped_pixmap:
            self.label_image.setPixmap(cropped_pixmap)
        else:
            self.label_image.setText("Žádný oříznutý obrázek")
        self.layout.addWidget(self.label_image)

        # Vstup pro zadání počtu clusterů
        label_clusters = QLabel("Počet clusterů:")
        self.layout.addWidget(label_clusters)
        self.input_clusters = QLineEdit()
        self.layout.addWidget(self.input_clusters)

        # Tlačítko pro vygenerování clusterů
        self.btn_generate_clusters = QPushButton("vygeneruj clustery")
        self.btn_generate_clusters.clicked.connect(self.on_generate_clusters)
        self.layout.addWidget(self.btn_generate_clusters)

        # Scroll area pro výsledné obrázky (serie obrázků)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.scroll_area.setWidget(self.results_container)
        self.layout.addWidget(self.scroll_area)

    def on_generate_clusters(self):
        if self.cropped_pixmap is None:
            error_label = QLabel("Není k dispozici oříznutý obrázek.")
            self.results_layout.addWidget(error_label)
            return

        # Uložíme QPixmap do dočasného souboru
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_sample = tmp_file.name
        if not self.cropped_pixmap.save(temp_sample):
            error_label = QLabel("Chyba při ukládání oříznutého obrázku.")
            self.results_layout.addWidget(error_label)
            return

        # Přečtení počtu clusterů
        try:
            cluster_count = int(self.input_clusters.text())
        except ValueError:
            error_label = QLabel("Zadejte platné číslo pro počet clusterů.")
            self.results_layout.addWidget(error_label)
            return

        # Vyprázdnit předchozí výsledky
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Zavoláme funkci, která zpracuje obrázek a vrátí seznam cest k výsledným obrázkům.
        image_paths = check_clusters_embedded(cluster_count, temp_sample)
        # Předpokládáme, že první dva obrázky nejsou clusterové (kontrast stretching, přeclusterovaný obrázek)
        cluster_image_paths = image_paths[2:] if len(image_paths) > 2 else image_paths

        # Pro každý cluster vytvoříme tlačítko s obrázkem jako ikonu
        for path in cluster_image_paths:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue
            button = QPushButton()
            button.setIcon(QIcon(pixmap))
            button.setIconSize(pixmap.size())
            button.setFlat(True)
            button.clicked.connect(partial(self.select_cluster, pixmap))
            self.results_layout.addWidget(button)
    # def select_cluster(self, pixmap):
    #     """Při výběru clusteru nastaví vybraný obrázek do cílového widgetu a zavře okno."""
    #     if self.target_label:
    #         self.target_label.setPixmap(pixmap)
    #     self.close()
    def select_cluster(self, pixmap):
        """Při výběru clusteru nastaví vybraný obrázek do cílového widgetu,
        uloží jej do instance full_quality_cropped a zavře okno."""
        if self.target_label:
            self.target_label.setPixmap(pixmap)
            main_window = self.target_label.window()
            if hasattr(main_window, 'full_quality_cropped'):
                main_window.full_quality_cropped = pixmap
        self.close()
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
        self.selection_anchor = None  # Nový atribut pro výběr oblasti s klávesou Shift
        self.drawing = False  # Indikuje, zda probíhá kreslení výběru
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # Aby widget přijímal klávesové události
        self.magnifier = MagnifierLabel()
        # Atributy pro režim kříže (pravítko)
        self.show_crosshair = False
        self.current_cursor_pos = QPoint(0, 0)
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
            # Pokud držíme Shift, použijeme aktuální pozici jako anchor
            if event.modifiers() & Qt.ShiftModifier:
                self.selection_anchor = event.pos()
            else:
                # Jinak resetujeme anchor a nastavíme nový počáteční bod
                self.selection_anchor = None
                self.start_point = event.pos()
            # self.start_point = event.pos()
            self.end_point = self.start_point
            self.selection_rect = QRect(self.start_point, self.end_point)
            self.drawing = True
            self.updateMagnifier(event)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.current_cursor_pos = event.pos()
        if self.drawing:
            # Pokud držíme Shift, použijeme uložený anchor pro výpočet výběru
            if event.modifiers() & Qt.ShiftModifier and self.selection_anchor is not None:
                self.end_point = event.pos()
                self.selection_rect = QRect(self.selection_anchor, self.end_point).normalized()
            else:
                # Jinak (bez Shift) používáme start_point (nastavený při kliknutí)
                self.end_point = event.pos()
                self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.updateMagnifier(event)
        else:
            # Pokud se jen pohybujeme, aktualizujeme lupu
            self.updateMagnifier(event)
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.drawing = False
            self.update()

    def keyPressEvent(self, event):
        step = 1  # standardní posun v pixelech
        new_pos = QPoint(self.current_cursor_pos)
        ctrl_pressed = event.modifiers() & Qt.ControlModifier
        shift_pressed = event.modifiers() & Qt.ShiftModifier

        if event.key() == Qt.Key_Left:
            if ctrl_pressed:
                new_pos = self.getNextBoundaryPos(self.current_cursor_pos, "left")
            else:
                new_pos -= QPoint(step, 0)
        elif event.key() == Qt.Key_Right:
            if ctrl_pressed:
                new_pos = self.getNextBoundaryPos(self.current_cursor_pos, "right")
            else:
                new_pos += QPoint(step, 0)
        elif event.key() == Qt.Key_Up:
            if ctrl_pressed:
                new_pos = self.getNextBoundaryPos(self.current_cursor_pos, "up")
            else:
                new_pos -= QPoint(0, step)
        elif event.key() == Qt.Key_Down:
            if ctrl_pressed:
                new_pos = self.getNextBoundaryPos(self.current_cursor_pos, "down")
            else:
                new_pos += QPoint(0, step)
        else:
            super().keyPressEvent(event)
            return

        # Pokud držíme Shift, nastavíme nebo využijeme anchor pro výběr
        if shift_pressed:
            if self.selection_anchor is None:
                self.selection_anchor = QPoint(self.current_cursor_pos)
            self.selection_rect = QRect(self.selection_anchor, new_pos).normalized()
        else:
            # Když Shift není stisknutý, pokud právě neprobíhá kreslení myší, anchor vymažeme
            if not self.drawing:
                self.selection_anchor = None
            if self.drawing:
                self.selection_rect = QRect(self.start_point, new_pos).normalized()

        self.current_cursor_pos = new_pos
        self.updateMagnifierAtPos(new_pos)
        self.update()
        event.accept()
    def getNextBoundaryPos(self, pos, direction):
        """
        Vrátí novou pozici kurzoru (ve widgetu), která odpovídá dalšímu místu změny barvy pixelu.
        """
        base_pixmap = self.pixmap()
        if base_pixmap is None:
            return pos

        # Přepočítej pozici widget -> pixmap
        label_width = self.width()
        label_height = self.height()
        displayed_width = base_pixmap.width()
        displayed_height = base_pixmap.height()
        offset_x = (label_width - displayed_width) // 2
        offset_y = (label_height - displayed_height) // 2
        pixmap_pos = pos - QPoint(offset_x, offset_y)

        # Získej QImage pro přístup k pixelům
        image = base_pixmap.toImage()
        # Zajisti, že pozice je v rozsahu
        x = max(0, min(pixmap_pos.x(), displayed_width - 1))
        y = max(0, min(pixmap_pos.y(), displayed_height - 1))
        current_color = image.pixel(x, y)
        new_pixmap_pos = QPoint(x, y)

        if direction == "down":
            for new_y in range(y + 1, displayed_height):
                if image.pixel(x, new_y) != current_color:
                    new_pixmap_pos.setY(new_y)
                    break
                new_pixmap_pos.setY(new_y)
        elif direction == "up":
            for new_y in range(y - 1, -1, -1):
                if image.pixel(x, new_y) != current_color:
                    new_pixmap_pos.setY(new_y)
                    break
                new_pixmap_pos.setY(new_y)
        elif direction == "right":
            for new_x in range(x + 1, displayed_width):
                if image.pixel(new_x, y) != current_color:
                    new_pixmap_pos.setX(new_x)
                    break
                new_pixmap_pos.setX(new_x)
        elif direction == "left":
            for new_x in range(x - 1, -1, -1):
                if image.pixel(new_x, y) != current_color:
                    new_pixmap_pos.setX(new_x)
                    break
                new_pixmap_pos.setX(new_x)

        # Přepočítáme zpět na souřadnice widgetu
        new_widget_pos = new_pixmap_pos + QPoint(offset_x, offset_y)
        return new_widget_pos
    def updateMagnifierAtPos(self, pos):
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

        # Pozice relativně k pixmapě
        pixmap_pos = pos - QPoint(offset_x, offset_y)

        region_size = 30
        half = region_size // 2

        # Podobná logika jako v updateMagnifier (s vlastními úpravami, pokud je kurzor mimo oblast)
        region_pixmap = QPixmap(region_size, region_size)
        region_pixmap.fill(Qt.transparent)
        painter = QPainter(region_pixmap)
        src_x = pixmap_pos.x() - half
        src_y = pixmap_pos.y() - half
        src_rect = QRect(src_x, src_y, region_size, region_size)
        dest_rect = QRect(0, 0, region_size, region_size)

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

        painter.drawPixmap(dest_rect, base_pixmap, src_rect)
        painter.end()

        magnified = region_pixmap.scaled(self.magnifier.width(), self.magnifier.height(),
                                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.magnifier.setMagnifiedPixmap(magnified)
        global_pos = self.mapToGlobal(pos)
        offset = 20
        self.magnifier.move(global_pos.x() + offset, global_pos.y() + offset)
        self.magnifier.show()
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


# class EraserImageWindow(QWidget):
#     def __init__(self, cropped_pixmap=None, target_label=None):
#         super().__init__()
#         self.setWindowTitle("Eraser Window")
#         self.cropped_pixmap = cropped_pixmap  # Originální pixmapa (full quality)
#         self.target_label = target_label  # Reference na widget, kde se má výsledný obrázek zobrazit
#         self.zoom_factor = 1.0  # Počáteční zoom faktor
#         self.init_ui(cropped_pixmap)
#         self.showMaximized()
#
#     def init_ui(self, cropped_pixmap):
#         self.layout = QVBoxLayout(self)
#         if cropped_pixmap and not cropped_pixmap.isNull():
#             # Vytvoříme widget Canvas pro úpravu obrázku
#             self.canvas = Canvas(cropped_pixmap, self)
#             # Vložíme Canvas do QScrollArea, aby bylo možné obrázek posouvat (bez automatického škálování)
#             scroll_area = QScrollArea()
#             scroll_area.setWidget(self.canvas)
#             self.layout.addWidget(scroll_area)
#         else:
#             self.label_image = QLabel("Žádný oříznutý obrázek")
#             self.label_image.setAlignment(Qt.AlignCenter)
#             self.layout.addWidget(self.label_image)
#
#         # Tlačítko pro uložení upraveného obrázku
#         self.btn_save = QPushButton("Uložit obrázek")
#         self.btn_save.clicked.connect(self.select_eraser)
#         self.layout.addWidget(self.btn_save)
#
#     def select_eraser(self):
#         """
#         Při stisknutí tlačítka:
#          - Převede aktuální QImage z Canvasu na QPixmap.
#          - Nastaví tento pixmap do target_label.
#          - Aktualizuje instanci, kde je uložen původní oříznutý obrázek (např. v MainWindow.full_quality_cropped),
#            a tím přepíše původní obrázek, který je vstupem pro funkci zpracovat spektrum.
#          - Zavře okno.
#         """
#         if self.target_label and hasattr(self, 'canvas'):
#             new_pixmap = QPixmap.fromImage(self.canvas.image)
#             self.target_label.setPixmap(new_pixmap)
#             # Aktualizace instance obsahující původní oříznutý obrázek.
#             # Předpokládáme, že target_label je součástí hlavního okna, kde je uložen atribut full_quality_cropped.
#             main_window = self.target_label.window()
#             if hasattr(main_window, 'full_quality_cropped'):
#                 main_window.full_quality_cropped = new_pixmap
#         self.close()

# class Canvas(QWidget):
#     def __init__(self, pixmap, parent=None):
#         super().__init__(parent)
#         self.eraserRadius = 10  # Poloměr štětce/eraseru
#         # Převod QPixmap na QImage v původní kvalitě (s podporou průhlednosti)
#         if pixmap is None or pixmap.isNull():
#             self.image = QImage()
#         else:
#             self.image = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
#         # Nastavení pevné velikosti dle obrázku
#         self.setFixedSize(self.image.size())
#         # Výchozí barva – pokud není vybrána, používá se eraser (clear)
#         self.brush_color = None
#
#     def mousePressEvent(self, event: QMouseEvent):
#         if event.button() == Qt.RightButton:
#             # Pravým tlačítkem otevřeme dialog pro výběr barvy
#             color = QColorDialog.getColor()
#             if color.isValid():
#                 self.brush_color = color
#         super().mousePressEvent(event)
#
#     def mouseMoveEvent(self, event: QMouseEvent):
#         if event.buttons() & Qt.LeftButton:
#             painter = QPainter(self.image)
#             painter.setPen(Qt.NoPen)
#             if self.brush_color is not None:
#                 # Použijeme vybranou barvu – režim kreslení přes (source over)
#                 painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
#                 painter.setBrush(self.brush_color)
#             else:
#                 # Pokud není barva vybrána, použijeme eraser (vymaže oblast)
#                 painter.setCompositionMode(QPainter.CompositionMode_Clear)
#                 painter.setBrush(Qt.black)  # Hodnota brush nehraje roli v Clear módu
#             # Kreslíme vyplněný kruh přímo pod kurzorem
#             painter.drawEllipse(event.pos(), self.eraserRadius, self.eraserRadius)
#             painter.end()
#             self.update()
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.drawImage(0, 0, self.image)
class EraserImageWindow(QWidget):
    def __init__(self, cropped_pixmap=None, target_label=None):
        super().__init__()
        self.setWindowTitle("Eraser Window")
        self.cropped_pixmap = cropped_pixmap  # Originální pixmapa (full quality)
        self.target_label = target_label  # Widget, kam se má výsledný obrázek uložit
        self.zoom_factor = 1.0  # Počáteční zoom faktor (není přímo použit v této třídě)
        self.init_ui(cropped_pixmap)
        self.showMaximized()

    def init_ui(self, cropped_pixmap):
        self.layout = QVBoxLayout(self)
        if cropped_pixmap and not cropped_pixmap.isNull():
            # Vytvoříme widget Canvas pro úpravu obrázku se zoomem a výběrem barvy
            self.canvas = Canvas(cropped_pixmap, self)
            # Vložíme Canvas do QScrollArea a centrování nastavíme pomocí setAlignment
            scroll_area = QScrollArea()
            scroll_area.setWidget(self.canvas)
            scroll_area.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(scroll_area)
        else:
            self.label_image = QLabel("Žádný oříznutý obrázek")
            self.label_image.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label_image)

        # Tlačítko pro uložení upraveného obrázku
        self.btn_save = QPushButton("Uložit obrázek")
        self.btn_save.clicked.connect(self.select_eraser)
        self.layout.addWidget(self.btn_save)

    def select_eraser(self):
        """
        Při stisknutí tlačítka:
         - Převádí aktuální QImage z Canvasu na QPixmap.
         - Nastaví tento pixmap do target_label.
         - Aktualizuje instanci, kde je uložen původní oříznutý obrázek,
           a tím přepíše původní obrázek, který je vstupem pro další zpracování.
         - Zavře okno.
        """
        if self.target_label and hasattr(self, 'canvas'):
            new_pixmap = QPixmap.fromImage(self.canvas.image)
            self.target_label.setPixmap(new_pixmap)
            # Aktualizace instance s původním obrázkem (např. MainWindow.full_quality_cropped)
            main_window = self.target_label.window()
            if hasattr(main_window, 'full_quality_cropped'):
                main_window.full_quality_cropped = new_pixmap
        self.close()
class Canvas(QWidget):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.eraserRadius = 10  # Poloměr štětce/eraseru
        # Převod QPixmap na QImage ve full quality se zachováním alfa kanálu
        if pixmap is None or pixmap.isNull():
            self.image = QImage()
        else:
            self.image = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        self.original_size = self.image.size()  # Původní velikost obrázku
        self.zoom_factor = 1.0  # Výchozí zoom faktor
        self.setFixedSize(self.original_size)
        # Výchozí barva pro kreslení; pokud není vybrána, použije se režim gumy
        self.brush_color = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        # Aplikujeme škálovací transformaci
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.drawImage(0, 0, self.image)

    def wheelEvent(self, event: QWheelEvent):
        # Zoom pomocí kolečka myši
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor *= 0.9
        # Nastavíme limity zoomu
        if self.zoom_factor < 0.1:
            self.zoom_factor = 0.1
        if self.zoom_factor > 10:
            self.zoom_factor = 10
        # Aktualizujeme velikost widgetu podle nového zoom faktoru
        new_width = int(self.original_size.width() * self.zoom_factor)
        new_height = int(self.original_size.height() * self.zoom_factor)
        self.setFixedSize(new_width, new_height)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            # Pravým tlačítkem vybereme barvu přímo z obrázku (color picker)
            # Přepočítáme souřadnice z widgetu na souřadnice originálního obrázku
            pos = event.pos()
            x = int(pos.x() / self.zoom_factor)
            y = int(pos.y() / self.zoom_factor)
            if x < self.image.width() and y < self.image.height():
                color = self.image.pixelColor(x, y)
                self.brush_color = color
                print("Vybraná barva:", color.name())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            self.paintAt(event)

    def paintAt(self, event: QMouseEvent):
        # Přepočítáme pozici z widget souřadnic na originální souřadnice obrázku
        pos = event.pos()
        x = int(pos.x() / self.zoom_factor)
        y = int(pos.y() / self.zoom_factor)
        painter = QPainter(self.image)
        painter.setPen(Qt.NoPen)
        if self.brush_color is not None:
            # Pokud máme vybranou barvu, kreslíme s ní (normální režim)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setBrush(self.brush_color)
        else:
            # Jinak použijeme režim gumy (clear)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setBrush(Qt.black)  # V Clear módu hodnota brush nehraje roli
        # Vykreslíme vyplněný kruh se středem v (x,y)
        painter.drawEllipse(QPoint(x, y), self.eraserRadius, self.eraserRadius)
        painter.end()
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow")
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
        label_xmin = QLabel("Xleft:")
        label_xmin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(label_xmin)

        self.input_xmin = QLineEdit()
        self.input_xmin.setFixedWidth(50)
        self.input_xmin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.input_xmin.setText("4000")  # Nastavení výchozí hodnoty
        param_layout.addWidget(self.input_xmin)

        # Xmax
        label_xmax = QLabel("Xright:")
        label_xmax.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(label_xmax)

        self.input_xmax = QLineEdit()
        self.input_xmax.setFixedWidth(50)
        self.input_xmax.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.input_xmax.setText("0")  # Nastavení výchozí hodnoty
        param_layout.addWidget(self.input_xmax)

        # Ymin
        label_ymin = QLabel("Ymin:")
        label_ymin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(label_ymin)

        self.input_ymin = QLineEdit()
        self.input_ymin.setFixedWidth(50)
        self.input_ymin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.input_ymin.setText("0")  # Nastavení výchozí hodnoty
        param_layout.addWidget(self.input_ymin)

        # Ymax
        label_ymax = QLabel("Ymax:")
        label_ymax.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        param_layout.addWidget(label_ymax)

        self.input_ymax = QLineEdit()
        self.input_ymax.setFixedWidth(50)
        self.input_ymax.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.input_ymax.setText("100")  # Nastavení výchozí hodnoty
        param_layout.addWidget(self.input_ymax)

        # Find peaks
        sensitivity_label = QLabel("Sensitivity:")
        self.input_sensitivity = QLineEdit()
        self.input_sensitivity.setFixedWidth(50)
        self.input_sensitivity.setText("10")  # Nastavení výchozí hodnoty
        min_distance_label = QLabel("Min distance:")
        self.input_min_distance = QLineEdit()
        self.input_min_distance.setFixedWidth(50)
        self.input_min_distance.setText("10")  # Nastavení výchozí hodnoty
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

        # Přidání nového tlačítka "Rozdělit na clustery"
        btn_cluster = QPushButton("Rozdělit na clustery")
        btn_cluster.clicked.connect(self.open_cluster_window)
        main_layout.addWidget(btn_cluster)

        # Tlačítko pro otevření okna s Eraser obrázkem
        btn_show_eraser = QPushButton("Zobrazit Eraser")
        btn_show_eraser.clicked.connect(self.openEraserImageWindow)
        right_layout.addWidget(btn_show_eraser)

        # Zabalíme celý central_widget do QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)

    # def openEraserImageWindow(self):
    #     # Získáme pixmapu z labelu, pokud byla nastavena
    #     pixmap = self.label_cropped.pixmap()
    #     if pixmap is None:
    #         QMessageBox.information(self, "Informace", "Není k dispozici žádný oříznutý obrázek!")
    #         return
    #
    #     self.eraser_window = EraserImageWindow(pixmap, self.label_cropped)
    #     # Okno se zobrazí automaticky díky showMaximized() v konstruktoru
    def openEraserImageWindow(self):
        if not hasattr(self, 'full_quality_cropped') or self.full_quality_cropped is None:
            QMessageBox.information(self, "Informace", "Nejdříve proveďte oříznutí obrázku!")
            return

        self.eraser_window = EraserImageWindow(self.full_quality_cropped, self.label_cropped)
        # Okno se zobrazí automaticky, pokud je v konstruktoru voláno showMaximized()
    # def open_cluster_window(self):
    #     cropped_pixmap = self.label_cropped.pixmap()
    #     self.cluster_window = ClusterWindow(cropped_pixmap, self.label_cropped)
    #     self.cluster_window.show()
    def open_cluster_window(self):
        if not hasattr(self, 'full_quality_cropped') or self.full_quality_cropped is None:
            QMessageBox.information(self, "Informace", "Nejdříve proveďte oříznutí obrázku!")
            return

        self.cluster_window = ClusterWindow(self.full_quality_cropped, self.label_cropped)
        self.cluster_window.show()
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

            # Uložíme plnou kvalitu oříznutého obrázku pro další zpracování
            self.full_quality_cropped = cropped_pixmap

            # Pro zobrazení v hlavním okně použijeme škálovanou verzi,
            # např. omezíme výšku na 300 pixelů
            max_display_height = 300
            display_pixmap = cropped_pixmap.scaledToHeight(max_display_height, Qt.SmoothTransformation)
            self.label_cropped.setPixmap(display_pixmap)

            # self.label_cropped.setPixmap(cropped_pixmap)

            # self.label_cropped.setPixmap(
            #     cropped_pixmap.scaled(self.label_cropped.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # )
        else:
            print("Obrázek nebyl načten nebo nebyla vybrána oblast!")

    def toggle_crosshair(self, checked):
        self.label_original.show_crosshair = checked
        self.label_original.update()

    # def process_cropped_image(self):
    #     """
    #     1) Zkontroluje, zda máme v label_cropped oříznutý obrázek.
    #     2) Převede QPixmap na numpy pole pomocí qpixmap_to_array.
    #     3) Načte hodnoty x_min, x_max, y_min, y_max z QLineEdit.
    #     4) Zavolá funkci preprocess_image_from_array (importovanou ze simple_line.py),
    #        která vrátí NumPy pole obrázku a hlavní konturu.
    #     5) Vykreslí graf spektra pomocí extract_and_plot_contour (také ze simple_line.py)
    #        a uloží výsledek do paměti.
    #     6) Výsledný graf zobrazí v label_result.
    #     """
    #     cropped_pixmap = self.label_cropped.pixmap()
    #     if not cropped_pixmap:
    #         # self.label_result.setText("Chybí oříznutý obrázek!")
    #         QMessageBox.warning(self, "Chyba", "Chybí oříznutý obrázek!")
    #         return
    #
    #     try:
    #         # Načtení hodnot z QLineEdit
    #         x_min = float(self.input_xmin.text())
    #         x_max = float(self.input_xmax.text())
    #         y_min = float(self.input_ymin.text())
    #         y_max = float(self.input_ymax.text())
    #     except ValueError:
    #         # self.label_result.setText("Chybné hodnoty Xmin/Xmax/Ymin/Ymax!")
    #         QMessageBox.warning(self, "Chyba", "Chybné hodnoty Xmin/Xmax/Ymin/Ymax!")
    #         return
    #
    #     try:
    #         # Vypneme interaktivní režim matplotlib
    #         plt.ioff()
    #
    #         # Převod QPixmap na numpy pole
    #         img_array = qpixmap_to_array(cropped_pixmap)
    #
    #         # Použijeme funkci preprocess_image_from_array, která očekává numpy pole
    #         # Ujisti se, že jsi tuto funkci importoval, např.:
    #         # from simple_line import preprocess_image_from_array, extract_and_plot_contour
    #         img, center_line, longest_contour = preprocess_image_from_array(img_array)
    #
    #         # Vykreslíme graf do matplotlibu (bez plt.show())
    #         # extract_and_plot_contour(img, main_contour, x_min, x_max, y_min, y_max)
    #
    #         # Vykreslíme graf a zároveň získáme data spektra (data_x, data_y)
    #         data_x, data_y = extract_and_plot_contour(img, center_line, x_min, x_max, y_min, y_max)
    #
    #         # Uložíme spektrum pro použití funkcí (např. v tlačítku "find peaks")
    #         self.last_x = data_x
    #         self.last_y = data_y
    #
    #         # Uložíme vykreslený graf do paměti
    #         buf = io.BytesIO()
    #         plt.savefig(buf, format='png')
    #         plt.close()  # Zavře figure
    #         buf.seek(0)
    #
    #         # Vytvoříme QImage z bytestreamu a nastavíme jej do label_result
    #         qimage = QImage.fromData(buf.getvalue(), 'PNG')
    #         if qimage.isNull():
    #             QMessageBox.critical(self, "Chyba", "Nepodařilo se vykreslit graf.")
    #             # self.label_result.setText("Nepodařilo se vykreslit graf.")
    #         else:
    #             pixmap = QPixmap.fromImage(qimage)
    #             # Omezíme maximální výšku výsledného grafu, aby se nezvětšoval vertikálně
    #             max_height = 600  # nastav dle potřeby
    #             if pixmap.height() > max_height:
    #                 pixmap = pixmap.scaledToHeight(max_height, Qt.SmoothTransformation)
    #             self.label_result.setPixmap(pixmap)
    #             self.statusBar().showMessage("Spektrum bylo úspěšně zpracováno.", 3000)
    #         # Zobrazíme popup s longest_contour
    #         self.show_longest_contour(longest_contour)
    #     except Exception as e:
    #         # self.label_result.setText(f"Nastala chyba při zpracování: {e}")
    #         QMessageBox.critical(self, "Chyba", f"Nastala chyba při zpracování: {e}")
    #     finally:
    #         plt.ioff()
    def process_cropped_image(self):
        """
        1) Zkontroluje, zda máme k dispozici full_quality_cropped.
        2) Převede QPixmap na numpy pole pomocí qpixmap_to_array.
        3) Načte hodnoty x_min, x_max, y_min, y_max z QLineEdit.
        4) Zavolá funkci preprocess_image_from_array (importovanou ze simple_line.py),
           která vrátí NumPy pole obrázku, centrální linii a nejdelší konturu.
        5) Vykreslí graf spektra pomocí extract_and_plot_contour (také ze simple_line.py)
           a uloží výsledek do paměti.
        6) Výsledný graf zobrazí v label_result.
        """
        # Použijeme obrázek v plné kvalitě namísto label_cropped.pixmap()
        if not hasattr(self, 'full_quality_cropped') or self.full_quality_cropped is None:
            QMessageBox.warning(self, "Chyba", "Chybí oříznutý obrázek ve full quality!")
            return

        cropped_pixmap = self.full_quality_cropped

        try:
            # Načtení hodnot z QLineEdit
            x_min = float(self.input_xmin.text())
            x_max = float(self.input_xmax.text())
            y_min = float(self.input_ymin.text())
            y_max = float(self.input_ymax.text())
        except ValueError:
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
            img, center_line, longest_contour = preprocess_image_from_array(img_array)

            # Vykreslíme graf a zároveň získáme data spektra (data_x, data_y)
            data_x, data_y = extract_and_plot_contour(img, center_line, x_min, x_max, y_min, y_max)

            # Uložíme spektrum pro další zpracování (např. v tlačítku "find peaks")
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
            else:
                pixmap = QPixmap.fromImage(qimage)
                # Omezíme maximální výšku výsledného grafu, aby se nezvětšoval vertikálně
                max_height = 600  # nastav dle potřeby
                if pixmap.height() > max_height:
                    pixmap = pixmap.scaledToHeight(max_height, Qt.SmoothTransformation)
                self.label_result.setPixmap(pixmap)
                self.statusBar().showMessage("Spektrum bylo úspěšně zpracováno.", 3000)

            # Zobrazíme popup s longest_contour
            self.show_longest_contour(longest_contour)
        except Exception as e:
            QMessageBox.critical(self, "Chyba", f"Nastala chyba při zpracování: {e}")
        finally:
            plt.ioff()
    def show_longest_contour(self, longest_contour):
        """
        Vytvoří a zobrazí vyskakovací okno s grafem longest_contour.
        """
        # Vytvoříme matplotlib figuru a vykreslíme longest_contour
        fig, ax = plt.subplots()
        ax.plot(longest_contour[:, 1], -longest_contour[:, 0], linewidth=2, label="Longest contour")
        ax.set_title("Longest contour")
        ax.legend()

        # Uložíme graf do paměti
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Převedeme uložený obrázek na QImage a QPixmap
        qimage = QImage.fromData(buf.getvalue(), 'PNG')
        pixmap = QPixmap.fromImage(qimage)

        # Vytvoříme dialog, kde zobrazíme QPixmap
        dialog = QDialog(self)
        dialog.setWindowTitle("Longest Contour")
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.exec_()

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
