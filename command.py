import os
from pathlib import Path

def check_dir(path, lst):
    direc = []
    for element in lst:
        temp = path + "/" + element
        if os.path.isdir(temp) == True:
            direc.append(temp)
    if not direc: return -1
    else: return direc
    
def check_file(lst): # check files and list up all cpp files in directory
    res = []      
    for element in lst:
        if element.endswith(".c"):
            res.append(element)
    if not res: return -1
    else: return res

def exec_command(lst, h): # execute command for extracting call graph
    for element in lst:
        fname = element.split('.c')[0]
        tmp = fname + '.h'
        if os.path.exists(tmp) == True:
            os.system(f'clang -v -include {h} -c -emit-llvm {element} -o {fname}')
            if os.path.exists(fname) == True:
                os.system(f'opt -dot-callgraph {fname}')
                os.system(f'mv callgraph.dot {fname}.dot')
                print("Generated (1) " + fname)
            else:
                print("<<<<< Failed to build executable file -- 1 >>>>>")
            #print(">>>>> now (1) : " + element)
        else:
            os.system(f'clang -v -c -emit-llvm {element} -o {fname}')
            if os.path.exists(fname) == True:
                os.system(f'opt -dot-callgraph {fname}')
                os.system(f'mv callgraph.dot {fname}.dot')
                print("Generated (2) " + fname)
            else:
                print("<<<<< Failed to build executable file -- 2 >>>>>")
            #print(">>>>> now (2) : " + element)
        #os.system(f'mv callgraph.dot {fname}.dot')
        
        tmp = fname + '.dot'
        if os.path.exists(tmp) == True:
            os.system(f'dot -Tjpg {fname}.dot -o {fname}.jpg')

def main():
    PATH = os.getcwd()
    
    FILE_LIST = os.listdir(PATH)
   
    DIR_LIST = check_dir(PATH, FILE_LIST) # check whether directories are here or not  
    #print(DIR_LIST[0])
    #path = Path(DIR_LIST[0])
    #print(path)

    header = []
    files = []
    for dir in DIR_LIST:
        repo = (dir.__str__())
        for file in Path(dir).rglob('*.h'):
            header.append(file.__str__())
        for file in Path(dir).rglob('*.c'):
            if not file.is_dir():
     #           print(file)
                files.append(file.__str__())
    #print(files)
    
    
    """
    if DIR_LIST == -1: # No directories in this path just check files of this path
        print("No directory here")
        FILE_LIST = check_file(FILE_LIST)
        if FILE_LIST == -1:
            print("No c++ files here")
            return
    else:
        for x in range(len(DIR_LIST)): # check files in directories
            temp = check_file(DIR_LIST[x])
            if temp == -1:
                print(f'No c++ files in {DIR_LIST[x]}')
            else:
                FILE_LIST = FILE_LIST + temp

        temp = check_file(FILE_LIST) # check files of this path
        if temp != -1:
            FILE_LIST = FILE_LIST + temp
        else:
            print("No c++ files here, terminate the program")
            return
     
    #print(FILE_LIST)
    #exec_command(FILE_LIST, header)
    """
    import subprocess

    for file in files:
        path = Path(file)
        cmd = ['clang', '-v', '-S', '-emit-llvm', path.__str__(), '-o', path.parent / path.stem]
        #print(path.__str__())
        #print(path.parent / path.stem)
        #for h in header:
            #cmd.append('-include')
            #cmd.append(h)
        subprocess.run(cmd, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        cmd = ['opt', '--dot-callgraph', path.parent / path.stem]
        subprocess.run(cmd)
        tmp = (path.parent / path.stem).__str__() + '.dot'
        cmd = ['mv', 'callgraph.dot', tmp]
        subprocess.run(cmd)
        tmp2 = (path.parent / path.stem).__str__()  + '.jpg'
        cmd = ['dot', '-Tjpg', tmp, '-o', tmp2]
        subprocess.run(cmd)



        


    return # finish

if __name__ == "__main__": # execute main function at first
    main()
