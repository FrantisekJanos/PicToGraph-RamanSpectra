import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import find_contours
import numpy as np
from PyQt5.QtGui import QImage
from collections import defaultdict



def qimage_to_array(qimage):
    qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    return arr[..., :3]  # Vrátíme jen RGB kanály

def contours_to_center_line(contours):
    """
    Sloučí všechny body z (jedné či více) kontur do jedné křivky.
    V každém integer sloupci x zprůměruje y-hodnoty a vytvoří 'středovou' linku.

    Parameters:
        contours (list of ndarray): List kontur z find_contours,
            každá kontura je Nx2 (y, x).

    Returns:
        center_line (ndarray): Pole tvaru (M, 2) s (y, x),
            kde pro každý integer x je jediné průměrné y.
    """
    x_dict = defaultdict(list)
    # Projdeme všechny kontury a všechny body v nich
    for contour in contours:
        for (y, x) in contour:
            x_int = int(round(x))  # zaokrouhlíme x na nejbližší celé číslo
            x_dict[x_int].append(y)

    # Vytvoříme výslednou křivku
    center_line = []
    for x_int in sorted(x_dict.keys()):
        ys = x_dict[x_int]
        # Zprůměrujeme všechny y v tomto sloupci
        y_mean = np.mean(ys)
        center_line.append([y_mean, x_int])

    return np.array(center_line)

def preprocess_image_from_array(img):
    """
    Načte obrázek, převede jej na stupně šedi, vytvoří binární masku pomocí Otsuova prahu,
    odstraní malé objekty a najde kontury v obrázku.

    Parameters:
        image_path (str): Cesta k obrázku.

    Returns:
        img (ndarray): Původní obrázek.
        main_contour (ndarray): Kontura s největší délkou.
    """
    # Předpokládáme, že img má alespoň 3 kanály (RGB)
    rgb = img[:, :, :3]  # Vybereme pouze RGB kanály
    gray = rgb2gray(rgb)  # Převedeme na stupně šedi

    # Aplikace Otsuova prahu pro binarizaci
    thresh = threshold_otsu(gray)
    binary = gray > thresh  # Předpokládáme, že křivka je tmavá

    # Odstranění malých šumů
    binary = remove_small_objects(binary, min_size=20)

    # Hledání kontur v binárním obrázku
    contours = find_contours(binary, level=0.5)

    if not contours:
        raise ValueError("Nebyla nalezena žádná kontura v obrázku.")
    longest_contour = max(contours, key=len)

    # # Vybereme konturu s největší délkou
    # main_contour = max(contours, key=len)
    center_line = contours_to_center_line([longest_contour])


    return img, center_line


def calculate_figsize(img, base_width=10):
    """
    Vypočítá hodnoty figsize pro matplotlib tak, aby byl zachován poměr stran vstupního obrázku.

    Parameters:
        img (ndarray): Původní obrázek jako NumPy pole.
        base_width (float, optional): Základní šířka v palcích pro figsize. Výchozí hodnota je 10.

    Returns:
        tuple: (šířka, výška) v palcích pro figsize.
    """
    if img.ndim < 2:
        raise ValueError("Obrázek musí mít alespoň 2 rozměry (výška a šířka).")

    height, width = img.shape[:2]

    if width == 0:
        raise ValueError("Šířka obrázku nemůže být nula.")

    aspect_ratio = height / width
    figsize = (base_width, base_width * aspect_ratio)

    return figsize


def extract_and_plot_contour(img, main_contour, x_min, x_max, y_min, y_max):
    """
    Extrahuje souřadnice kontury, transformuje je do reálných hodnot a vykreslí graf spektra.

    Parameters:
        img (ndarray): Původní obrázek.
        main_contour (ndarray): Kontura s největší délkou.
        x_min (float): Minimální hodnota na ose x.
        x_max (float): Maximální hodnota na ose x.
        y_min (float): Minimální hodnota na ose y.
        y_max (float): Maximální hodnota na ose y.
    """
    # Extrakce x, y souřadnic z kontury
    ys, xs = main_contour[:, 0], main_contour[:, 1]

    # Transformace z pixelů do reálných hodnot
    data_x = x_min + (xs / img.shape[1]) * (x_max - x_min)
    data_y = y_max - (ys / img.shape[0]) * (y_max - y_min)
    # Pokud je osa y obrácená, můžete upravit podle potřeby

    # Výpočet figsize pro zachování poměru stran
    figsize = calculate_figsize(img)

    # Vytvoření figure a axis objektů s dynamickým figsize
    plt.figure(figsize=figsize)

    # Vykreslení dat
    plt.plot(data_x, data_y, linestyle='-', color='b', label='Spektrum')

    # Přidání názvu a popisků os
    plt.title('Extrahovaný Graf Spektra')
    plt.xlabel('Vlnová délka (nm)')  # Upravte podle skutečných údajů
    plt.ylabel('Intenzita')  # Upravte podle skutečných údajů

    # Přidání mřížky
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Přidání legendy
    plt.legend()

    # Zobrazení grafu
    # plt.show()

    # Místo plt.show() vracíme data spektra
    return data_x, data_y

def main(sample_name):
    """
    Hlavní funkce programu. Definuje název obrázku, zavolá předzpracování a následné zpracování.
    """
    # Definujte název obrázku
    sample_name2 = f'{sample_name}'  # Změňte na skutečný název obrázku

    # Sestavte cestu k obrázku
    image_path = f'obrazkyspekter/{sample_name2}'

    # Vykreslení obrázku (i v případě výjimky)
    try:
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.title("Vstupní Obrázek")
        plt.axis('off')  # Skrytí os
        plt.show()
    except FileNotFoundError:
        print(f"Obrázek '{image_path}' nebyl nalezen pro zobrazení.")
    except Exception as e:
        print(f"Nastala chyba při vykreslování obrázku: {e}")

    try:
        # Předzpracování obrázku
        img, main_contour = preprocess_image(image_path)

        # Extrakce a vykreslení kontury
        extract_and_plot_contour(img, main_contour, x_min=0, x_max=4000, y_min=0, y_max=1000)
    except FileNotFoundError:
        print(f"Obrázek '{image_path}' nebyl nalezen.")
    except ValueError as ve:
        print(f"Chyba při zpracování obrázku: {ve}")
    except Exception as e:
        print(f"Nastala neočekávaná chyba: {e}")
        # Načtení a zobrazení obrázku
