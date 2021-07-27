import os

file_path = './image_urls.txt'
with open(file_path) as fp:
    line = fp.readline()
    count = 1
    while line:
        cmd = f'curl "{line.strip()}" >> ./images/{count}.png'
        os.system(cmd)
        count += 1
        line = fp.readline()
