import sys, os
import glob
import shutil
# from IPython.nbformat import current as nbformat
# from IPython.nbconvert import PythonExporter
import nbformat
from nbconvert import PythonExporter
import subprocess
import ast
from tqdm import tqdm


def scan(data_file):
    # note: I have checked os_file_listall, I think the following will be better
    files = glob.glob(data_file + '/**/*.py', recursive=True)
    # remove .ipynb_checkpoints
    files = [s for s in files if '.ipynb_checkpoints' not in s]

    print('scan files done ... ')
    return files



def get_packages(file):

    with open(file, "r") as fp :
        contents = fp.read()

    packages = []
    for line in contents.strip().split('\n'):
        line = line.strip()
        if not line: continue
        # if this is commented
        if line[0]=='#': continue
        if 'import ' not in line: continue


        package = None
        if 'from' in line:
            # from XX.XX import XX
            # from XX import XX
            line = line[line.index('from')+4:].strip()
            if '.' in line:
                package = line[:min(line.index('.'),line.index(' '))]
            else:
                package = line[:line.index(' ')]
        else:
            if '.' in line:
                # import XX.XX
                line = line[line.index('import')+6:].strip()
                package = line[:line.index('.')]
            else:
                # import XX
                line = line[line.index('import')+6:].strip()
                package = line

        # XX as XX
        if ' as ' in package:
            package = package[:package.index(' as ')]
        # XX ; XX
        if ';' in package:
            package = package[:package.index(';')].strip()
        # import XX,XX
        if ',' in package:
            for p in package.strip().split(','):
                packages.append(p.strip())
        else:
            packages.append(package)

    return packages



def get_missing(all_packages):
    # # form import XX,XX,XX
    # run_string = 'import '+ ','.join(all_packages)
    # # maybe cd to the dir

    miss_package = []


    for package in all_packages:
        run_string = 'import '+ package
        try:
            exec(run_string)
            # os.system('python -c "%s"'%run_string)
        except:
            miss_package.append(package)



    return miss_package





def Run():
    if len(sys.argv) != 2:
        print('Syntax: %s src_fold' % sys.argv[0])
        sys.exit(0)
    data_file = sys.argv[1]



    # scan file recursively
    source_files = scan(data_file)
    print(len(source_files))


    # check packages
    need_to_install_package_list = []
    for file in source_files:
        all_packages = get_packages(file)
        miss_packages = get_missing(all_packages)
        need_to_install_package_list.extend(miss_packages)

        # if len(miss_packages):
        #     print('-'*10, file)
        #     print(miss_packages)


    need_to_install_package_set = list(set(need_to_install_package_list))
    # print(need_to_install_package_set)
    # print(len(need_to_install_package_set))


    white_lists = ['resnet', 'mobilenet', 'inception', 'utils']
    need_to_install_package_set = [s for s in need_to_install_package_set if s]
    need_to_install_package_set = [s for s in need_to_install_package_set if not any([w in s for w in white_lists])]

    print(need_to_install_package_set)
    print(len(need_to_install_package_set))


    with open('./require_before.txt', 'w') as fp:
        fp.write('\n'.join(need_to_install_package_set))

    # auto install
    for package in need_to_install_package_set:
        # os.system('conda install %s'%package)
        os.system('pip install %s'%package)


    # check again
    miss_packages = get_missing(need_to_install_package_set)
    with open('./require_after.txt', 'w') as fp:
        fp.write('\n'.join(miss_packages))



if __name__ == '__main__':
    Run()
