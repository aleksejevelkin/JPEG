from PIL import Image
import numpy as np
from math import ceil

def image_to_raw_rgb(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def rgb_to_ycbcr_jpeg(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.clip([y, cb, cr], 0, 255).astype('uint8')


def convert_image_to_ycbcr(image_path):
    # Шаг 1: Загружаем изображение и получаем RAW RGB
    rgb_data = image_to_raw_rgb(image_path)
    height, width, _ = rgb_data.shape

    # Шаг 2: Конвертируем каждый пиксель в YCbCr
    ycbcr_data = np.zeros((height, width, 3), dtype='uint8')
    for y in range(height):
        for x in range(width):
            r, g, b = rgb_data[y, x]
            ycbcr_data[y, x] = rgb_to_ycbcr_jpeg(r, g, b)

    return ycbcr_data


def downsample_channel(channel, ratio=(2, 2)):
    h, w = channel.shape
    new_h = h // ratio[0]
    new_w = w // ratio[1]

    downsampled = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            block = channel[i * ratio[0]:(i + 1) * ratio[0],
                    j * ratio[1]:(j + 1) * ratio[1]]
            downsampled[i, j] = np.mean(block)

    return downsampled


def save_image(path, y, cb, cr):
    cb_upscaled = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1)
    cr_upscaled = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1)

    target_height, target_width = y.shape
    cb_upscaled = cb_upscaled[:target_height, :target_width]
    cr_upscaled = cr_upscaled[:target_height, :target_width]

    ycbcr = np.dstack((y, cb_upscaled, cr_upscaled)).astype('uint8')
    Image.fromarray(ycbcr, 'YCbCr').save(path)


def split_into_blocks_with_padding(y, cb, cr, original_shape, block_size: int = 8, padding_value: int = 0) -> dict:

    def _pad_and_split(channel):
        h, w = channel.shape
        # Вычисляем новые размеры с дополнением
        new_h = ceil(h / block_size) * block_size
        new_w = ceil(w / block_size) * block_size

        # Создаем массив с дополнением
        padded = np.full((new_h, new_w), padding_value, dtype=channel.dtype)
        padded[:h, :w] = channel  # Копируем оригинальные данные

        # Разбиваем на блоки
        blocks = padded.reshape(new_h // block_size, block_size,
                                new_w // block_size, block_size).transpose(0, 2, 1, 3)
        return blocks

    return {
        'y_blocks': _pad_and_split(y),
        'cb_blocks': _pad_and_split(cb),
        'cr_blocks': _pad_and_split(cr),
        'original_shape': original_shape
    }


def reconstruct_from_padded_blocks(blocks: np.ndarray, original_shape: tuple) -> np.ndarray:

    reconstructed = blocks.transpose(0, 2, 1, 3).reshape(
        blocks.shape[0] * blocks.shape[2],
        blocks.shape[1] * blocks.shape[3]
    )
    return reconstructed[:original_shape[0], :original_shape[1]]


import numpy as np

def get_dct_matrix(size=8):
    C = np.zeros((size, size))
    for u in range(size):
        for x in range(size):
            c_u = np.sqrt(1 / size) if u == 0 else np.sqrt(2 / size)
            C[u, x] = c_u * np.cos((2 * x + 1) * u * np.pi / (2 * size))
    return C

def dct2d(blocks: np.ndarray):
    C = get_dct_matrix(8)
    dct_coeffs = {
        'y_dct': np.zeros_like(blocks['y_blocks'], dtype=np.float32),
        'cb_dct': np.zeros_like(blocks['cb_blocks'], dtype=np.float32),
        'cr_dct': np.zeros_like(blocks['cr_blocks'], dtype=np.float32),
        'original_shape': blocks.get('original_shape', None)
    }

    channels_dct = ['y_dct', 'cb_dct', 'cr_dct']
    channels_original = ['y_blocks', 'cb_blocks', 'cr_blocks']

    for channel, original_name in zip(channels_dct, channels_original):
        for i in range(blocks[original_name].shape[0]):
            for j in range(blocks[original_name].shape[1]):
                block = blocks[original_name][i, j]
                dct_coeffs[channel][i, j] = C @ block @ C.T

    return dct_coeffs

def idct2d(dct_blocks: np.ndarray):
    C = get_dct_matrix(8)
    C_inv = C.T

    reconstructed = {
        'y_blocks': np.zeros_like(dct_blocks['y_dct']),
        'cb_blocks': np.zeros_like(dct_blocks['cb_dct']),
        'cr_blocks': np.zeros_like(dct_blocks['cr_dct']),
    }

    dct_channels = ['y_dct', 'cb_dct', 'cr_dct']
    idct_channels = ['y_blocks', 'cb_blocks', 'cr_blocks']

    for idct_channel, dct_channel in zip(idct_channels, dct_channels):
        for i in range(dct_blocks[dct_channel].shape[0]):
            for j in range(dct_blocks[dct_channel].shape[1]):
                block = dct_blocks[dct_channel][i, j]
                reconstructed[idct_channel][i, j] = C_inv @ block @ C_inv.T

    return reconstructed


def quant(dct_blocks, quality=50):
    Q_Y = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32)

    Q_C = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.float32)

    quantized = {
        'y_quant': np.zeros_like(dct_blocks['y_dct']),
        'cb_quant': np.zeros_like(dct_blocks['cb_dct']),
        'cr_quant': np.zeros_like(dct_blocks['cr_dct']),
        'original_shape': dct_blocks['original_shape']
    }

    # КВАНТОВАНИЕ Y

    for i in range(dct_blocks['y_dct'].shape[0]):
        for j in range(dct_blocks['y_dct'].shape[1]):

            for v in range(8):
                for u in range(8):

                    Q = 0

                    if quality > 50:
                        Q = Q_Y[v, u] * ((100 - quality) / 50)
                    if quality <= 50:
                        Q = Q_Y[v, u] * 50 / quality
                    if quality == 100:
                        quantized['y_quant'][i][j][v][u] = dct_blocks['y_dct'][i][j][v][u]
                        continue
                    quantized['y_quant'][i][j][v][u] = np.round(dct_blocks['y_dct'][i][j][v][u] / Q)

    # КВАНТОВАНИЕ Cb

    for i in range(dct_blocks['cb_dct'].shape[0]):
        for j in range(dct_blocks['cb_dct'].shape[1]):

            for v in range(8):
                for u in range(8):

                    Q = 0

                    if quality > 50:
                        Q = Q_C[v, u] * (100 - quality) / 50
                    if quality <= 50:
                        Q = Q_C[v, u] * 50 / quality
                    if quality == 100:
                        quantized['cb_quant'][i][j][v][u] = dct_blocks['cb_dct'][i][j][v][u]
                        continue

                    quantized['cb_quant'][i][j][v][u] = np.round(dct_blocks['cb_dct'][i][j][v][u] / Q)

    # КВАНТОВАНИЕ Cb

    for i in range(dct_blocks['cr_dct'].shape[0]):
        for j in range(dct_blocks['cr_dct'].shape[1]):

            for v in range(8):
                for u in range(8):

                    Q = 0

                    if quality > 50:
                        Q = Q_C[v, u] * (100 - quality) / 50
                    if quality <= 50:
                        Q = Q_C[v, u] * 50 / quality
                    if quality == 100:
                        quantized['cr_quant'][i][j][v][u] = dct_blocks['cr_dct'][i][j][v][u]
                        continue

                    quantized['cr_quant'][i][j][v][u] = np.round(dct_blocks['cr_dct'][i][j][v][u] / Q)
    return quantized


def dequant(quantized_blocks, original_shape, quality=50, ):
    Q_Y = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32)

    Q_C = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.float32)

    dequantized = {
        'y_dct': np.empty_like(quantized_blocks['y_quant'], dtype=np.float32),
        'cb_dct': np.empty_like(quantized_blocks['cb_quant'], dtype=np.float32),
        'cr_dct': np.empty_like(quantized_blocks['cr_quant'], dtype=np.float32),
        'original_shape': original_shape
    }

    # деКВАНТОВАНИЕ Y

    for i in range(quantized_blocks['y_quant'].shape[0]):
        for j in range(quantized_blocks['y_quant'].shape[1]):

            for v in range(8):
                for u in range(8):

                    Q = 0

                    if quality > 50:
                        Q = Q_Y[v, u] * (100 - quality) / 50
                    if quality <= 50:
                        Q = Q_Y[v, u] * 50 / quality
                    if quality == 100:
                        Q = 1


                    dequantized['y_dct'][i][j][v][u] = quantized_blocks['y_quant'][i][j][v][u] * Q

    # деКВАНТОВАНИЕ Cb

    for i in range(quantized_blocks['cb_quant'].shape[0]):
        for j in range(quantized_blocks['cb_quant'].shape[1]):

            for v in range(8):
                for u in range(8):

                    Q = 0

                    if quality > 50:
                        Q = Q_C[v, u] * (100 - quality) / 50
                    if quality <= 50:
                        Q = Q_C[v, u] * 50 / quality
                    if quality == 100:
                        Q = 1
                    dequantized['cb_dct'][i][j][v][u] = quantized_blocks['cb_quant'][i][j][v][u] * Q

    # деКВАНТОВАНИЕ Cb

    for i in range(quantized_blocks['cr_quant'].shape[0]):
        for j in range(quantized_blocks['cr_quant'].shape[1]):

            for v in range(8):
                for u in range(8):

                    Q = 0

                    if quality > 50:
                        Q = Q_C[v, u] * (100 - quality) / 50
                    if quality <= 50:
                        Q = Q_C[v, u] * 50 / quality
                    if quality == 100:
                        Q = 1
                    dequantized['cr_dct'][i][j][v][u] = quantized_blocks['cr_quant'][i][j][v][u] * Q
    return dequantized


zigzag_order = np.array([
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
])


def zigzag_scan_blocks(blocks):
    zigzag_blocks = {}
    for channel in ['y_quant', 'cb_quant', 'cr_quant']:
        h, w = blocks[channel].shape[:2]
        zigzag_blocks[channel] = np.empty((h, w, 64), dtype=np.int32)
        for i in range(h):
            for j in range(w):
                block = blocks[channel][i, j]
                zigzag_blocks[channel][i, j] = block[tuple(zigzag_order.T)]

    if 'original_shape' in blocks:
        zigzag_blocks['original_shape'] = blocks['original_shape']
    return zigzag_blocks


def inverse_zigzag_scan_blocks(zigzag_blocks):
    blocks = {}
    for channel in ['y_quant', 'cb_quant', 'cr_quant']:
        h, w = zigzag_blocks[channel].shape[:2]
        blocks[channel] = np.empty((h, w, 8, 8), dtype=np.int32)
        for i in range(h):
            for j in range(w):
                flat = zigzag_blocks[channel][i, j]
                block = np.empty((8, 8), dtype=np.int32)
                for idx, (x, y) in enumerate(zigzag_order):
                    block[x, y] = flat[idx]
                blocks[channel][i, j] = block

    if 'original_shape' in zigzag_blocks:
        blocks['original_shape'] = zigzag_blocks['original_shape']
    return blocks


def diff_encoding(zigzag_blocks):
    channels_quant = ['y_quant', 'cb_quant', 'cr_quant']
    encoded = {}

    for channel_q in channels_quant:
        prev = 0
        h, w = zigzag_blocks[channel_q].shape[:2]
        encoded[channel_q] = np.empty((h, w, 64), dtype=np.int32)

        for i in range(h):
            for j in range(w):
                current = zigzag_blocks[channel_q][i, j][0]
                encoded[channel_q][i, j] = zigzag_blocks[channel_q][i, j].copy()
                encoded[channel_q][i, j][0] = current - prev
                prev = current

    if 'original_shape' in zigzag_blocks:
        encoded['original_shape'] = zigzag_blocks['original_shape']

    return encoded


def diff_decoding(encoded_zigzag_blocks):
    channels_quant = ['y_quant', 'cb_quant', 'cr_quant']
    decoded = {}

    for channel_q in channels_quant:
        prev = 0
        encoded_block = np.array(encoded_zigzag_blocks[channel_q])
        h, w = encoded_block.shape[:2]
        # h, w = encoded_zigzag_blocks[channel_q].shape[:2]
        decoded[channel_q] = np.empty((h, w, 64), dtype=np.int32)

        for i in range(h):
            for j in range(w):
                decoded[channel_q][i][j] = encoded_zigzag_blocks[channel_q][i][j].copy()
                decoded[channel_q][i][j][0] += prev
                prev = decoded[channel_q][i][j][0]

    if 'original_shape' in encoded_zigzag_blocks:
        decoded['original_shape'] = encoded_zigzag_blocks['original_shape']

    return decoded


from huffman_table import *

def encode_magnitude(value, size):
    if value >= 0:
        return format(value, f'0{size}b')
    else:
        encoded = 2 ** size + value - 1
        return format(encoded, f'0{size}b')


def decode_magnitude(value_bits):
    if not value_bits:
        return 0
    if value_bits[0] == '1':
        # Положительное число
        return int(value_bits, 2)
    else:
        decoded = int(value_bits, 2) - (2 ** len(value_bits) - 1)
        return decoded

def func(array, channel):
    new_array = ""

    if (channel == 'y_quant'):
        dc = STD_LUMA_DC_HUFFMAN
        ac = STD_LUMA_AC_HUFFMAN
    else:
        dc = STD_CHROMA_DC_HUFFMAN
        ac = STD_CHROMA_AC_HUFFMAN


    DC_value = array[0]
    DC_size = int(abs(DC_value)).bit_length() if DC_value != 0 else 0

    DC_prefix = dc[DC_size]
    DC_bits = encode_magnitude(DC_value, DC_size) if DC_size > 0 else ""
    new_array += DC_prefix + DC_bits

    run = 0
    for i in range(1, len(array)):
        val = array[i]
        if val == 0:
            run += 1
            continue

        size = int(abs(val)).bit_length()
        while run > 15:
            new_array += ac[(15, 0)]  # ZRL (Zero Run Length)
            run -= 16

        AC_prefix = ac[(run, size)]
        AC_bits = encode_magnitude(val, size)
        new_array += AC_prefix + AC_bits
        run = 0

    # Добавляем EOB (End Of Block), если нужно
    if run > 0:
        new_array += ac[(0, 0)]  # EOB

    return new_array


def save_bits_to_file(bit_string: str, filename: str):
    """
    Упаковывает битовую строку в байты и сохраняет в файл

    :param bit_string: строка из битов ('010101...')
    :param filename: имя файла для сохранения
    """
    # Добавляем нули в конец, чтобы длина была кратна 8

    padding = 8 - (len(bit_string) % 8)
    if padding != 8:
        bit_string += '0' * padding

    # Преобразуем битовую строку в байты
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i + 8]
        byte_array.append(int(byte, 2))

    # Записываем в файл
    with open(filename, 'ab') as f:
        # Первый байт - количество добавленных бит (0-7)
        f.write(bytes([padding]))
        f.write(byte_array)



def load_bits_from_file(filename: str) -> str:

    with open(filename, 'rb') as f:
        temp = f.read(4)
        padding = ord(f.read(1))
        # Читаем остальные байты
        byte_data = f.read()

    # Преобразуем байты в битовую строку
    bit_string = ''.join(f'{byte:08b}' for byte in byte_data)

    # Удаляем добавленные нули
    if padding > 0:
        bit_string = bit_string[:-padding]

    return bit_string


def inverse_func(bit_string, marker1, channel):
    if (channel == 'y_quant'):
        dc = STD_LUMA_DC_HUFFMAN
        ac = STD_LUMA_AC_HUFFMAN
    else:
        dc = STD_CHROMA_DC_HUFFMAN
        ac = STD_CHROMA_AC_HUFFMAN

    array = []
    pos = 0

    elements = 1

    # Декодируем DC-компоненту
    for size, prefix in dc.items():
        if bit_string.startswith(prefix, pos):
            pos += len(prefix)
            marker1 += len(prefix)
            if size == 0:
                array.append(0)
            else:
                value_bits = bit_string[pos:pos + size]
                value = decode_magnitude(value_bits)
                array.append(value)
                pos += size
                marker1 += size
            break
    else:
        raise ValueError("Invalid DC code")

    while pos < len(bit_string) and elements < 64:
        for (run, size), prefix in ac.items():
            if bit_string.startswith(prefix, pos):
                pos += len(prefix)
                marker1 += len(prefix)

                # EOB
                if run == 0 and size == 0:
                    array.extend([0] * (64 - elements))
                    return array, marker1

                # Добавляем нули
                if elements + run > 64:
                    array.extend([0] * (64 - elements))
                    return array, marker1

                array.extend([0] * run)
                elements += run

                if size == 0:
                    array.append(0)
                    elements += 1
                else:
                    value_bits = bit_string[pos:pos + size]
                    value = decode_magnitude(value_bits)
                    array.append(value)
                    pos += size
                    marker1 += size
                    elements += 1
                break
        else:

            raise ValueError("Invalid AC code")

    # Заполняем оставшиеся нулями, если EOB не встретился
    if elements < 64:
        array.extend([0] * (64 - elements))

    return array, marker1


def save(fp, blocks):
    channels = ['y_quant', 'cb_quant', 'cr_quant']

    bits = ''

    with open(fp, 'wb') as f:
        f.write(blocks['original_shape'][0].to_bytes(2, byteorder='little'))
        f.write(blocks['original_shape'][1].to_bytes(2, byteorder='little'))
        f.close()

    for ch in channels:
        for i in range(blocks[ch].shape[0]):
            for j in range(blocks[ch].shape[1]):
                bits += func(blocks[ch][i][j], ch)

    save_bits_to_file(bits, fp)


def read(fp):
    blocks = {}

    with open(fp, 'rb') as f:
        height = int.from_bytes(f.read(2), byteorder='little')
        width = int.from_bytes(f.read(2), byteorder='little')

    Y_height = ceil((height / 8))
    Y_width = ceil(width / 8)

    CbCr_height = ceil(height / 16)
    CbCr_width = ceil(width / 16)

    blocks['y_quant'] = np.zeros((Y_height, Y_width, 64), dtype=np.int32)
    blocks['cb_quant'] = np.zeros((CbCr_height, CbCr_width, 64), dtype=np.int32)
    blocks['cr_quant'] = np.zeros((CbCr_height, CbCr_width, 64), dtype=np.int32)



    all_data = load_bits_from_file(fp)

    marker1 = 0

    for i in range(Y_height):
        for j in range(Y_width):
            data = all_data[marker1:]

            restored_array, marker1 = inverse_func(data, marker1, 'y_quant')
            if len(restored_array) != 64:
                restored_array = restored_array[:64]  # Обрезаем лишнее
            blocks['y_quant'][i][j] = restored_array

    for i in range(CbCr_height):
        for j in range(CbCr_width):
            data = all_data[marker1:]

            restored_array, marker1 = inverse_func(data, marker1, 'cb')
            if len(restored_array) != 64:
                restored_array = restored_array[:64]  # Обрезаем лишнее
            blocks['cb_quant'][i][j] = restored_array

    for i in range(CbCr_height):
        for j in range(CbCr_width):
            data = all_data[marker1:]

            restored_array, marker1 = inverse_func(data, marker1, 'cr')
            if len(restored_array) != 64:
                restored_array = restored_array[:64]  # Обрезаем лишнее
            blocks['cr_quant'][i][j] = restored_array

    return blocks


# Пример использования

input_image = "data/Lenna.png"
#input_image = "data/bird.png"
output_image = "data/output_decompressed.jpg"

# Конвертация в YCbCr
ycbcr = convert_image_to_ycbcr(input_image)

original_shape = ycbcr.shape

# Разделение на каналы
y = ycbcr[:, :, 0].astype(float)  # Y-канал (не даунсэмплится)
cb = ycbcr[:, :, 1].astype(float)  # Cb-канал
cr = ycbcr[:, :, 2].astype(float)  # Cr-канал

# Даунсэмплинг цветностных каналов
cb_down = downsample_channel(cb, ratio=(2, 2))
cr_down = downsample_channel(cr, ratio=(2, 2))

blocks = split_into_blocks_with_padding(y, cb_down, cr_down, original_shape)

# блоки - [вертикаль][горизонталь][вертикаль в блоке][горизонталь в блоке]


dct_coeffs = dct2d(blocks)  # DCT

quantized = quant(dct_coeffs, quality=80)  # Квантование

zigzag_blocks = zigzag_scan_blocks(quantized)  # Зигзаг

DCencoded = diff_encoding(zigzag_blocks)  # Разностное кодирование DC

######################################

save('bits.bin', DCencoded)
DCencoded = read('bits.bin')

######################################
decoded_blocks = diff_decoding(DCencoded)  # Разностное декодирование DC

decoded_blocks = inverse_zigzag_scan_blocks(decoded_blocks)  # обратный зигзаг

dequantized = dequant(decoded_blocks, original_shape, quality=80)  # деквантование

reconstructed_blocks = idct2d(dequantized)

y_reconstructed = reconstruct_from_padded_blocks(reconstructed_blocks['y_blocks'], original_shape)
cb_reconstructed = reconstruct_from_padded_blocks(reconstructed_blocks['cb_blocks'], original_shape)
cr_reconstructed = reconstruct_from_padded_blocks(reconstructed_blocks['cr_blocks'], original_shape)

# Сохранение результата
save_image(output_image, y_reconstructed, cb_reconstructed, cr_reconstructed)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('data/output_decompressed.jpg')  # Загрузка изображения
plt.imshow(img)
plt.axis('off')
plt.show()

