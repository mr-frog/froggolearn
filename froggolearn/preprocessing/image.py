from PIL import Image

def make_square(im, size, fill_color=(0, 0, 0, 0)):
    """
    Resizes a given image so its longer size becomes the specified size,
    while the shorter side is scaled correspondingly. The picture is then
    pasted onto a square shape with fill_color filling the empty space.
    """
    x, y = im.size
    long = np.max([x, y])
    x_new = int(x*size/long)
    y_new = int(y*size/long)
    im = im.resize((x_new, y_new), Image.ANTIALIAS)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im)
    return new_im
