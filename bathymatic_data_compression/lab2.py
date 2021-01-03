import pandas as pd
from scipy.fftpack import dct, idct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm


def compress(grid, size_block, padding_h, padding_w, coords):
    h, w = grid.shape
    if h % size_block != 0:
        padding_h_zeros = np.zeros((h, padding_h))
        padding_h_zeros[:] = np.nan
        grid = np.hstack([grid, padding_h_zeros])
    if w % size_block != 0:
        padding_w_zeros = np.zeros((padding_w, w + padding_h))
        padding_w_zeros[:] = np.nan
        grid = np.vstack([grid, padding_w_zeros])
    h, w = grid.shape
    compression = np.zeros(grid.shape, dtype='float32')
    grid_ar = np.array(grid)
    for i in tqdm.tqdm(range(0, h, size_block), desc=coords):
        for j in range(0, w, size_block):
            compression[i:i+size_block, j:j+size_block] = dct(grid_ar[i:i+size_block, j:j+size_block])
    return compression


def decompress(compressed, size_block, padding_h, padding_w, coords):
    decompression = np.zeros(compressed.shape, dtype='float32')
    h, w = compressed.shape
    for i in tqdm.tqdm(range(0, h, size_block), desc=coords):
        for j in range(0, w, size_block):
            decompression[i:i + size_block, j:j + size_block] = idct(compressed[i:i + size_block, j:j + size_block])
    if padding_h != 0 and padding_w != 0:
        decompression_new = decompression[:-padding_w, :-padding_h]
    elif padding_h != 0 and padding_w == 0:
        decompression_new = decompression[:-padding_w, :]
    elif padding_h == 0 and padding_w != 0:
        decompression_new = decompression[:, :-padding_h]
    else:
        decompression_new = decompression[:, :]
    return decompression_new


def get_meshgrids(data):
    x_lim, y_lim = get_limits(data[:, 0], data[:, 1])
    xx, yy = np.meshgrid(np.arange(x_lim[0], x_lim[1], x_lim[2]), np.arange(y_lim[0], y_lim[1], y_lim[2]))
    zz = np.reshape(data[:, 2], yy.shape)
    return xx, yy, zz


def get_limits(x, y):
    x_min = np.min(x)
    x_max = np.max(np.round(x, 2))
    step = x[1] - x[0]
    y_min = np.min(y)
    y_max = np.max(np.round(y, 2))
    return (x_min, x_max, step), (y_min, y_max, step)


def main():
    df = pd.read_csv("wraki_samochodów_outout_5.csv", names=["x", "y", "z"], sep='\s+')

    size_block = int(input("Block size: "))
    # size_block = 10

    acc = float(input("Accuracy: "))
    # acc = 0.05

    to_zip = input("To zip?[Y/N]: ")
    to_zip = False if to_zip =='N' else True

    data = df.to_numpy()
    xx, yy, zz = get_meshgrids(data)

    padding_h_x = size_block - (xx.shape[0] % size_block)
    padding_w_x = size_block - (xx.shape[1] % size_block)
    xx_comp = compress(xx, size_block, padding_h_x, padding_w_x, 'X compression')
    padding_h_y = size_block - (yy.shape[0] % size_block)
    padding_w_y = size_block - (yy.shape[1] % size_block)
    yy_comp = compress(yy, size_block, padding_h_y, padding_w_y, 'Y compression')
    padding_h_z = size_block - (zz.shape[0] % size_block)
    padding_w_z = size_block - (zz.shape[1] % size_block)
    zz_comp = compress(zz, size_block, padding_h_z, padding_w_z, 'Z compression')

    xx_decomp = decompress(xx_comp, size_block, padding_h_x, padding_w_x, 'X decompression')
    yy_decomp = decompress(yy_comp, size_block, padding_h_y, padding_w_y, 'Y decompression')
    zz_decomp = decompress(zz_comp, size_block, padding_h_z, padding_w_z, 'Z decompression')

    z_abs = abs(zz - zz_decomp)

    df_kom = pd.DataFrame({'x': xx_comp.flatten(), 'y': yy_comp.flatten(), 'z': zz_comp.flatten()})
    if to_zip:
        df_kom.to_csv('wraki_COMP.csv.gz', compression='gzip', sep=' ', header=False, index=False)
        cr_value = Path('wraki_samochodów_outout_5.csv').stat().st_size / Path('wraki_COMP.csv.gz').stat().st_size
        print(f'Compression ratio CSV: {cr_value}')
    else:
        df_kom.to_csv('wraki_COMP.csv', sep=' ', header=False, index=False)
        cr_value = Path('wraki_samochodów_outout_5.csv').stat().st_size / Path('wraki_COMP.csv').stat().st_size
        print(f'Compression ratio ZIP: {cr_value}')

    plt.figure()
    plt.title('Decompressed')
    p = plt.imshow(zz_decomp)
    plt.colorbar(p)

    plt.figure()
    plt.title('Compressed')
    p = plt.imshow(zz_comp)
    plt.colorbar(p)

    plt.figure()
    plt.title('Original')
    p = plt.imshow(zz)
    plt.colorbar(p)
    print(f'z_absz_absz_abs{z_abs}')
    plt.figure()
    plt.title('Errors')
    p = plt.imshow(z_abs, cmap='magma')
    plt.colorbar(p)

    plt.show()


if __name__ == '__main__':
    main()
