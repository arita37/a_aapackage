import logging
import os

import pytest

from aapackage import util_log


def os_path_append(f):
    assert isinstance(f, str)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f)


def test_create_uniqueid():
    assert len(util_log.create_uniqueid()) == len("_YYYYMMDDHHmmss1000_")


def test_create_logfilename():
    assert util_log.create_logfilename("test") == "test.log"
    assert util_log.create_logfilename("/home/test") == "test.log"
    assert util_log.create_logfilename("C:\\User/user1/test") == "test.log"


def test_printlog():
    assert len(util_log.APP_ID) != 0


def test_writelog():
    assert len(util_log.LOG_FILE) != 0

    fileout = os_path_append("mockup_test_util.log2")
    util_log.writelog("testing", fileout)
    with open(fileout, "r") as f:
        assert f.readline() == "testing\n"

    # Cleanup
    os.remove(fileout)


def test_logger_setup(capsys):
    # Test : console output / file output

    formatter = logging.Formatter("%(levelname)s %(message)s")
    fileout = os_path_append("mockup_test_util.log")
    logger = util_log.logger_setup("test_util_log", fileout, formatter)
    logger.info("mockup_test_util_log")

    output = "INFO mockup_test_util_log\n"

    # Console output
    out, err = capsys.readouterr()
    assert out == output

    # File output
    with open(fileout, "r") as f:
        assert f.readline() == output

    # Cleanup
    os.remove(fileout)


def test_logger_setup2(capsys):
    # Test : console output

    logger = util_log.logger_setup2()
    logger.info("mockup_test_util_log")

    out, err = capsys.readouterr()
    assert err == "mockup_test_util_log\n"
