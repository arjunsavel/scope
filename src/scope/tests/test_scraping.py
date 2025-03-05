from scope.input_output import read_crires_data
from scope.scrape_igrins_etc import *


def test_scrape_igrins():
    res = scrape_igrins_etc(8, 120)
    assert res == "100"


# now need to test the ... crires+ etc scraper.


def test_scrape_crires_plus():
    kmag = 8
    stellar_teff = 5000
    stellar_logg = 4.5
    exposure_time = 120
    res = scrape_crires_plus_etc(kmag, stellar_teff, stellar_logg, exposure_time)
    n_orders, n_wavs, wl_grid, snr_grid = read_crires_data("output.json")
    assert (
        n_orders > 0
        and n_wavs > 0
        and len(wl_grid) > 0
        and len(snr_grid) > 0
        and np.mean(snr_grid) > 0
    )
