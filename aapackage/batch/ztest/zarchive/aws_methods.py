#!/bin/python

"""


"""

import os

from Crypto.PublicKey import RSA

from ansible.executor.playbook_executor import PlaybookExecutor
from ansible.inventory.manager import Inventory
from ansible.parsing.dataloader import DataLoader


def generatePrivateKey():
    key = RSA.generate(2048)

    # Write Private key;
    PrivateFilePath = "private.key"
    with open(PrivateFilePath, "wb") as content_file:
        content_file.write(key.exportKey("PEM"))

    os.chmod(PrivateFilePath, 600)
    # Write Public Key;
    pubkey = key.publickey()
    with open("public.key", "wb") as content_file:
        content_file.write(pubkey.exportKey("OpenSSH"))


def setupRemoteInstance(InventoryPath):
    loader = DataLoader()
    inventory = Inventory(loader=loader, sources=InventoryPath)
