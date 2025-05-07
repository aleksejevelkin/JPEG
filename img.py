from PIL import Image, ImageOps

def convert_to_grayscale(image_path: str, output_path: str = None) -> Image.Image:
    """
    Конвертирует изображение в градации серого (grayscale).

    :param image_path: Путь к исходному изображению.
    :param output_path: Путь для сохранения (если None, изображение не сохраняется).
    :return: Объект PIL.Image в градациях серого.
    """
    img = Image.open(image_path).convert("L")
    if output_path:
        img.save(output_path)
    return img


def convert_to_bw_no_dithering(image_path: str, output_path: str = None, threshold: int = 128) -> Image.Image:
    """
    Конвертирует изображение в ЧБ (без дизеринга), используя пороговое значение.

    :param image_path: Путь к исходному изображению.
    :param output_path: Путь для сохранения (если None, изображение не сохраняется).
    :param threshold: Порог бинаризации (0-255).
    :return: Объект PIL.Image в ЧБ.
    """
    img = Image.open(image_path).convert("L")
    # Применяем пороговую бинаризацию
    img_bw = img.point(lambda p: 255 if p > threshold else 0)
    if output_path:
        img_bw.save(output_path)
    return img_bw


def convert_to_bw_with_dithering(image_path: str, output_path: str = None,
                                 method: str = "floyd-steinberg") -> Image.Image:
    """
    Конвертирует изображение в ЧБ с дизерингом.

    :param image_path: Путь к исходному изображению.
    :param output_path: Путь для сохранения (если None, изображение не сохраняется).
    :param method: Метод дизеринга ("floyd-steinberg", "bayer", "atkinson").
    :return: Объект PIL.Image с дизерингом.
    """
    img = Image.open(image_path).convert("L")

    if method == "floyd-steinberg":
        # Дизеринг Флойда-Стейнберга (встроенный в Pillow)
        img_bw = img.convert("1")

    if output_path:
        img_bw.save(output_path)
    return img_bw


# Пример использования:
if __name__ == "__main__":
    input_image = "data/bird.png"

    # Градации серого
    gray_img = convert_to_grayscale(input_image, "data/output/bird_grayscale.png")

    # ЧБ без дизеринга (порог 128)
    bw_img = convert_to_bw_no_dithering(input_image, "data/output/bird_bw_without_dith.png")

    # ЧБ с дизерингом (Флойд-Стейнберг)
    dither_img = convert_to_bw_with_dithering(input_image, "data/output/bird_bw_with_dith.png", method="floyd-steinberg")