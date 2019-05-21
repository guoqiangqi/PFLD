import os
from PIL import Image,ImageSequence
import imageio
import os
import shutil

if __name__ == "__main__":

    inf = r'C:\Users\Administrator\Desktop\picacho\gif/img-9.gif'
    outf = r'C:\Users\Administrator\Desktop\picacho\gif\img-9-d.gif'

    gif = Image.open(inf)
    dura = gif.info['duration']
    imgs = [f.copy() for f in ImageSequence.Iterator(gif)]

    index = 0
    imglist = []
    os.mkdir("images")
    for frame in imgs:
        frame.save("./images/%d.png" % index)
        im = Image.open("./images/%d.png" % index)
        im.thumbnail((200, 200), Image.ANTIALIAS)
        imglist.append(im)
        index += 1

    shutil.rmtree('./images')

    imglist[0].save(outf, 'gif', save_all=True, append_images=imglist[1:], loop=0, duration=dura)
