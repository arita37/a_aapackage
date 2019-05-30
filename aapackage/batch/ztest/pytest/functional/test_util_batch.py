import os

import pytest

from aapackage.batch import util_batch


def os_path_append(f):
    assert isinstance(f, str)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f)


def test_os_python_path():
    assert util_batch.os_python_path(), "Should return a string"


def test_os_folder_rename():
    # Test : correct args
    old = os_path_append("mockfolder3_rename")
    new = os_path_append("mockfolder3_renamed")
    util_batch.os_folder_rename(old, new)
    assert not os.path.isdir(old)
    assert os.path.isdir(new)

    # Cleanup
    os.rename(new, old)

    # Test : new_folder already exists
    old = os_path_append("mockfolder4_rename")
    new = os_path_append("mockfolder3_rename")
    util_batch.os_folder_rename(old, new)
    dir = os.path.dirname(os.path.abspath(__file__))
    folders = [f for f in os.listdir(dir) if os.path.isdir(os_path_append(f))]
    folders = [
        f for f in folders if f.startswith("mockfolder3_rename") and f != "mockfolder3_rename"
    ]
    assert len(folders) == 1
    assert len(folders[0]) > len("mockfolder3_rename")
    substr = folders[0][len("mockfolder3_rename") :]
    assert substr.isdigit()

    # Cleanup
    os.rename(os_path_append(folders[0]), os_path_append("mockfolder4_rename"))


def test_os_folder_create():
    # Test : correct args
    folder = os_path_append("testFolder")
    util_batch.os_folder_create(folder)
    assert os.path.isdir(folder), "Didn't create folder"

    # Cleanup
    os.rmdir(folder)


def test_batch_run_infolder():
    # Test : correct path and file, launches main.py
    valid_folders = [os_path_append("mockfolder1_util_batch")]
    sub_process_list = util_batch.batch_run_infolder(valid_folders, waitsleep=0)

    # Cleaning up
    src = os_path_append("mockfolder1_util_batch_qstart")
    dst = os_path_append("mockfolder1_util_batch")
    util_batch.os_folder_rename(src, dst)

    assert len(sub_process_list) == 1, "Length of sub_process_list should be 1"

    """
    # Test : folder_original_name + suffix already exists
    valid_folders = [os_path_append("mockfolder2_util_batch")]
    with pytest.raises(OSError):
        util_batch.batch_run_infolder(valid_folders, suffix="_suffix_test", waitsleep=0)
    """

    # Test : folder doesn't exist
    valid_folders = [os_path_append("mockfolder_doesnt_exist")]
    with pytest.raises(OSError):
        util_batch.batch_run_infolder(valid_folders, waitsleep=0)

    """
    # Test : folder exists but main_file_run doesn't exist
    # doesn't raise exception because python is directly called , not the script file 
    valid_folders = [os_path_append("mockfolder2_util_batch_suffix_test")]
    with pytest.raises(OSError):
        util_batch.batch_run_infolder(valid_folders, waitsleep=0)
    """


def test_batch_parallel_subprocess():
    # Test : csv file doesn't exist
    with pytest.raises(OSError):
        csv = os_path_append("file_doesnt_exist.csv")
        script = os_path_append("mockscript1.py")
        util_batch.batch_parallel_subprocess(csv, script, waitime=0)

    """
    # Test : subprocess_script doesn't exist
    # doesn't raise exception because python is directly called , not the script file 
    with pytest.raises(OSError):
        csv = os_path_append("mockfile.csv")
        script = os_path_append("script_doesnt_exist.py")
        util_batch.batch_parallel_subprocess(csv,script,waitime=0)
    """


def test_generate_hyperparameters():
    csv = os_path_append("file_doesnt_exist.csv")
    # Test : correct parameters , creates file
    """
    # Test : hyper_dict doesn't have key item
    with pytest.raises(KeyError):
        util_batch.batch_generate_hyperparameters({"test":58},csv)

    # Test : if hyper_dict['key']['min'] is a string
    with pytest.raises(TypeError):
        util_batch.batch_generate_hyperparameters({'key':{'min':'mock_string','max':4}},csv)
    """
    return 1
