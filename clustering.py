import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans
from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk
from skimage.measure import find_contours
import svgwrite
import os

def preprocess_image(image_path):
    """
    Načte obrázek, odstraní alfa kanál (pokud existuje), a škáluje hodnoty pixelů na rozsah 0-255.
    """
    # Načtení obrázku
    img = mpimg.imread(image_path)

    # Kontrola a odstranění alpha kanálu
    if img.shape[2] == 4:
        img_rgb = img[:, :, :3]
        alpha = img[:, :, 3]
        # Premultiplied alpha: vynásobení RGB hodnot alfa kanálem
        img_rgb = img_rgb * alpha[:, :, np.newaxis]
        print("Obrázek obsahuje Alpha kanál a byl odstraněn.")
    else:
        img_rgb = img
        print("Obrázek neobsahuje Alpha kanál.")

    # Kontrola datového typu a škálování na 0-255, pokud je to nutné
    if img_rgb.dtype in [np.float32, np.float64]:
        if img_rgb.max() <= 1.0:
            img_rgb_scaled = (img_rgb * 255).astype(np.uint8)
            print("RGB hodnoty byly škálovány na 0-255 a převedeny na uint8.")
        else:
            img_rgb_scaled = img_rgb.astype(np.uint8)
            print("RGB hodnoty byly převedeny na uint8 bez škálování.")
    else:
        img_rgb_scaled = img_rgb
        print("RGB hodnoty již jsou ve formátu uint8.")

    return img_rgb_scaled


def increase_contrast(img_scaled):
    """
    Zvýší kontrast obrázku pomocí contrast stretching.
    """
    # Definice rozsahu na základě percentilů
    p2, p98 = np.percentile(img_scaled, (2, 98))
    img_stretched = exposure.rescale_intensity(img_scaled, in_range=(p2, p98))

    # Převod zpět na uint8
    img_stretched = img_stretched.astype(np.uint8)

    print("Kontrast obrázku byl zvýšen pomocí contrast stretching.")
    return img_stretched


def cluster_colors(img_stretched, k=4):
    """
    Aplikuje KMeans klastrování na přeclusterovaný obrázek.
    """
    # Přetvoření obrázku do dvourozměrného pole pro KMeans
    height, width, channels = img_stretched.shape
    img_reshaped = img_stretched.reshape((-1, 3))
    print(f"Reshaped data shape for KMeans: {img_reshaped.shape}")

    # Inicializace a trénink KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img_reshaped)

    # Získání centroidů a labelů
    centroids = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    print(f"KMeans klastrování dokončeno. Počet klastrů: {k}")

    # Vytvoření přeclusterovaného obrázku
    img_clustered = centroids[labels].reshape((height, width, 3))

    return img_clustered, labels


def display_clusters(img_clustered, k=4):
    """
    Zobrazí přeclusterovaný obrázek s různými klustry.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img_clustered)
    plt.axis('off')
    plt.title(f'Obrázek s {k} Barvami po KMeans Klastrování')
    plt.show()


def compute_cluster_ratio(img):
    """
    Najde nejmenší obdélník (bounding box), který obsahuje všechny
    'ne-bílé' pixely (tj. pixely klastru) v obrázku `img`.
    Z rozměrů bounding boxu vypočítá poměr (šířka / výška).

    Parametry:
    -----------
    img : numpy.ndarray
        Obrázek, kde "pozadí" je bílé [255, 255, 255] a zbytek jsou
        pixely klastru, který nás zajímá.

    Návratová hodnota:
    ------------------
    ratio : float
        Poměr šířka / výška bounding boxu klastru.
    """
    # Vytvoř masku, která bude True pro všechny pixely != bílá
    mask = np.any(img != [255, 255, 255], axis=-1)
    coords = np.argwhere(mask)

    if coords.size == 0:
        print("Ve vybraném obrázku nebyly nalezeny žádné pixely klastru.")
        return 1.0  # Defaultní návrat, abychom se vyhnuli dělení nulou

    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    height = y2 - y1 + 1
    width = x2 - x1 + 1

    ratio = width / height
    return ratio


def get_figsize_for_cluster(img, base_size=8):
    """
    Na základě vypočteného poměru (šířka/výška) vrací rozměry figsize
    (width, height) tak, aby měla nejdelší strana velikost base_size.

    Parametry:
    -----------
    img : numpy.ndarray
        Obrázek, kde bílé pixely = [255,255,255], tj. ty, co nechceme započítávat.
    base_size : float, default 8
        Velikost (v palcích), která se použije jako "nejdelší" strana.

    Návratová hodnota:
    ------------------
    (fig_w, fig_h) : tuple(float, float)
        Doporučené hodnoty pro figsize (šířka, výška), které uchovají
        poměr stran klastru.
    """
    # Spočítáme poměr klastru (width / height)
    ratio = compute_cluster_ratio(img)  # Správně volá img, ne img_selected

    if ratio >= 1.0:
        # Širší než vyšší (nebo čtverec)
        # -> width = base_size, height = base_size / ratio
        fig_w = base_size
        fig_h = base_size / ratio if ratio != 0 else base_size
    else:
        # Vyšší než širší
        # -> height = base_size, width = base_size * ratio
        fig_h = base_size
        fig_w = base_size * ratio

    return (fig_w, fig_h)


def display_selected_cluster(img_clustered, labels, cluster_index):
    """
    Vytvoří obrázek, kde jsou viditelné pouze pixely z vybraného klastru.

    Returns:
    - img_selected: Obrázek s pouze vybraným klastrem.
    """
    height, width, _ = img_clustered.shape
    mask = (labels == cluster_index)
    mask_image = mask.reshape((height, width))


    # Vytvoření obrázku s pouze vybraným klastrem
    img_selected = img_clustered.copy()
    img_selected[~mask_image] = [255, 255, 255]  # Můžete změnit na jinou barvu

    # Zobrazení původního přeclusterovaného obrázku a obrázku s vybraným klastrem
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Přeclusterovaný obrázek
    axes[0].imshow(img_clustered)
    axes[0].axis('off')
    axes[0].set_title('Přeclusterovaný Obrázek')

    # Obrázek s pouze vybraným klastrem
    axes[1].imshow(img_selected)
    axes[1].axis('off')
    axes[1].set_title(f'Obrázek pouze s Cluster {cluster_index}')

    plt.show()

    return img_selected


def check_clusters(cluster_count, sample_name):
    # Parametry
    image_path = sample_name
    k = cluster_count  # Počet klastrů
    # selected_cluster = 3            # Klastr, který chcete extrahovat


    # Krok 1: Načtení a předzpracování obrázku
    img_scaled = preprocess_image(image_path)
    # Krok 2: Zvýšení kontrastu
    img_stretched = increase_contrast(img_scaled)

    # Zobrazení obrázku s roztaženým kontrastem
    plt.figure(figsize=(8, 6))
    plt.imshow(img_stretched)
    plt.axis('off')
    plt.title('Obrázek s Contrast Stretching')
    plt.show()
    # Krok 3: Klastrování
    img_clustered, labels = cluster_colors(img_stretched, k=k)

    # Krok 4: Zobrazení přeclusterovaného obrázku
    display_clusters(img_clustered, k=k)

    # Krok 5: Zobrazení pouze vybraného klastru
    for n in range(k):
        img_selected = display_selected_cluster(img_clustered, labels, n)