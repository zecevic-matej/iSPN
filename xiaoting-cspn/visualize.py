import visdom

vis = visdom.Visdom()

def draw_image(img_arr):
    mode = 'L' if len(img_arr.shape) == 2 else 'RGB'
    img = Image.fromarray((img_arr * 255).astype(np.uint8), mode=mode)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    rectangles = np.zeros((z_where.shape[0], 4, 2))
    for i in range(z_where.shape[0]):
        if z_pres[i] > 0.3:
            rectangles[i] = bounding_box(z_where[i], window_size)
            # corners = list(map(tuple, corners))
            # color = tuple(map(lambda c: int(c * z_pres[i]), (255, 255, 255)))
            # draw.polygon(corners, outline=color)
    # draw.text((0, 0), text, fill='red')
    draw_rectangles_with_overlap(draw, rectangles, z_pres)
    return img2arr(img)


