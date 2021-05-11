import os

def decompress(d):
    tmp = os.listdir(d)
    for element in tmp:
        if element.endswith("rar"):
            tmp = element.split('.rar')[0]
            os.system(f'mkdir {tmp}')
           os.system(f'cd {tmp}')
            os.system(f'unrar e {element}')
            os.system(f'cd ..')
        elif element.endswith("7z"):
            tmp = element.split('.7z')[0]
            os.system(f'7zr x {element}')
        elif element.endswith("zip"):
            tmp = element.split(".zip")[0]
            os.system(f'mkdir {tmp}')
            os.system(f'cd {tmp}')
            os.system(f'unzip {element}')
            os.system(f'cd ..')

def path(p):
    tmp = os.listdir(p) # get directory list in this path
    d = [(p + '/' + element) for element in tmp if os.path.isdir(p + '/' + element)]
    d.append(p) # add current path
    return d

def main():
    p = os.getcwd()
    d = path(p)

    for element in d:
        decompress(element)

    return
    
if __name__ == "__main__":
    main()
