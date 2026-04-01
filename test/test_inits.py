import pandas as pd
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from Carlsten_Satellites.dwarfs import Dwarf


#def test_dwarfs():
#    result = Dwarf(name ='DW0846P3314', dist_mod = 29.63, logmass = 6.65, Host = 'NGC 2683')
#    assert isinstance(result, Dwarf)
#
#def test_load_data():
#    d = Dwarf(name ='DW0846P3314', dist_mod = 29.63, logmass = 6.65, Host = 'NGC 2683')
#    d.load_data()
#    assert d.data is not None
#
#def test_load_ast():
#    d = Dwarf(name ='DW0846P3314', dist_mod = 29.63, logmass = 6.65, Host = 'NGC 2683')
#    d.load_asts()
#    assert d.asts is not None

def test_dwarfs():
    d = Dwarf('DW0846P3314', 29.63, 6.65, 'NGC 2683')
    assert isinstance(d, Dwarf)


def test_load_data(monkeypatch):
    def fake_read_hdf(*args, **kwargs):
        return pd.DataFrame({"ra": [1], "dec": [1]})

    monkeypatch.setattr("Carlsten_Satellites.dwarfs.pd.read_hdf", fake_read_hdf)

    d = Dwarf('DW0846P3314', 29.63, 6.65, 'NGC 2683')
    d.load_data()

    assert isinstance(d.data, pd.DataFrame)


def test_load_asts(monkeypatch):
    def fake_ascii_read(*args, **kwargs):
        return Table({"col1": [1]})

    monkeypatch.setattr("Carlsten_Satellites.dwarfs.ascii.read", fake_ascii_read)

    d = Dwarf('DW0846P3314', 29.63, 6.65, 'NGC 2683')
    d.load_asts()

    assert isinstance(d.asts, Table)
