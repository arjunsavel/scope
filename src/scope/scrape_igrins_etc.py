"""
pings the IGRINS exposure time calculator.


includes functions from  etc_cli.py, available on the ESO website: e.g., https://etc.eso.org/crires2
"""
import json
import os
import re
import sys
import urllib.parse

import numpy as np
import requests
from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager

from scope.logger import *

logger = get_logger()


def scrape_igrins_etc(kmag, exposure_time):
    """
    pings the IGRINS exposure time calculator.

    Parameters
    ----------
    kmag : float
        K-band magnitude of the target.
    exptime : float
        Exposure time in seconds.

    Returns
    -------
    float
        Exposure time in seconds.
    """
    options = Options()
    options.add_argument("--headless")  # Run in headless mode

    # Use WebDriver Manager to handle ChromeDriver installation
    service = Service(ChromeDriverManager().install())

    # Start Chrome WebDriver with options
    driver = webdriver.Chrome(service=service, options=options)

    # Start Edge WebDriver with options

    web_address = "https://igrins-jj.firebaseapp.com/etc/simple"

    driver.get(web_address)

    try:
        input_fields = driver.find_elements(By.TAG_NAME, "input")
        # first is kmag, second is SNR, third is kmag, fourth is integration time
        for i, input_field in enumerate(input_fields):
            input_field.clear()
            if i in [0, 2]:
                input_field.send_keys(str(kmag))
            if i == 3:
                input_field.send_keys(str(exposure_time))
        response_page = driver.page_source

    # now need to grab the exposure time at the end

    except ElementNotInteractableException:
        logger.error("Could not interact with IGRINS SNR website.")

    snr_value = extract_snr_from_html(response_page)

    return snr_value


def extract_snr_from_html(html_content):
    """
    Extracts the SNR value from the HTML content of the IGRINS SNR calculator.

    """
    # Regular expression to match the number following the comment structure for SNR
    pattern = r"<!-- react-text: 43 -->\s*(\d+)\s*<!-- /react-text -->"
    match = re.search(pattern, html_content)
    if match:
        snr_value = match.group(1)
        logger.info(f"Extracted SNR value: {snr_value}")
    else:
        logger.warning("SNR value not found.")
    return snr_value


# todo: automatically get the orders? or only allow one settingkey?


def get_closest_marcs_teff(teff):
    marcs_grid = np.concatenate(
        [np.arange(3000, 4100, 100), np.arange(4000, 8250, 250)]
    )
    return marcs_grid[
        np.argmin(np.abs(marcs_grid - teff))
    ]  # return the closest value in the grid


def get_closest_marcs_logg(logg):
    marcs_grid = np.arange(3, 5.5, 0.5)
    return marcs_grid[
        np.argmin(np.abs(marcs_grid - logg))
    ]  # return the closest value in the grid


# for now, only K band.
def scrape_crires_plus_etc(kmag, stellar_teff, stellar_logg, exposure_time):
    marcs_teff = get_closest_marcs_teff(stellar_teff)
    marcs_logg = get_closest_marcs_logg(stellar_logg)
    marcs_id = f"{marcs_teff}:{marcs_logg}"
    create_json(marcs_id, kmag, [23, 24, 25, 26, 27, 28, 29], exposure_time, "K2148")

    # now ping the CRIRES+ ETC
    main_etc_caller("input.json", "output.json")

    return


def create_json(
    marcs_id,
    target_brightness_mag,
    instrument_order,
    timesnr_DIT,
    instrument_settingkey,
    sky_airmass=1.0,
    sky_moon_fli=0.5,
    sky_pwv=2.0,
    instrument_slit=0.2,
    seeingiqao_turbulence_category=50,
    seeingiqao_aperturepix=17,
):
    data = {
        "target": {
            "morphology": {"morphologytype": "point"},
            "sed": {
                "sedtype": "spectrum",
                "spectrum": {
                    "spectrumtype": "template",
                    "params": {
                        "catalog": "MARCS",
                        "id": marcs_id,
                    },
                },
                "extinctionav": 0,
            },
            "brightness": {
                "brightnesstype": "mag",
                "magband": "V",
                "mag": target_brightness_mag,
                "magsys": "vega",
            },
        },
        "sky": {"airmass": sky_airmass, "moon_fli": sky_moon_fli, "pwv": sky_pwv},
        "instrument": {
            "slit": instrument_slit,
            "settingkey": instrument_settingkey,
            "polarimetry": "free",
            "order": instrument_order,
        },
        "timesnr": {"DET1.NDIT": 1, "DET1.DIT": timesnr_DIT},
        "output": {
            "throughput": {
                "atmosphere": False,
                "telescope": False,
                "instrument": False,
                "blaze": False,
                "enslittedenergy": False,
                "detector": False,
                "totalinclsky": False,
            },
            "snr": {"snr": True, "noise_components": True},
            "sed": {"target": False, "sky": False},
            "signals": {"obstarget": False, "obssky": False, "obstotal": False},
            "maxsignals": {
                "maxpixeltarget": False,
                "maxpixelsky": False,
                "maxpixeltotal": False,
            },
            "dispersion": {"dispersion": False},
            "psf": {"psf": False},
        },
        "instrumentName": "crires2",
        "seeingiqao": {
            "mode": "noao",
            "params": {"turbulence_category": seeingiqao_turbulence_category},
            "aperturepix": seeingiqao_aperturepix,
        },
    }

    # Writing the JSON to a file
    with open("input.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    logger.info("JSON file 'output.json' has been created.")


def collapse(jsondata):
    def goThrough(x):
        if isinstance(x, list):
            return goThroughList(x)
        elif isinstance(x, dict):
            return goThroughDict(x)
        else:
            return x

    def goThroughDict(dic):
        for key, value in dic.items():
            if isinstance(value, dict):
                dic[key] = goThroughDict(value)
            elif isinstance(value, list):
                dic[key] = goThroughList(value)
        return dic

    def goThroughList(lst):
        if not any(not isinstance(y, (int, float)) for y in lst):  # pure numeric list
            if len(lst) <= 2:
                return lst
            else:
                return (
                    "["
                    + str(lst[0])
                    + " ... "
                    + str(lst[-1])
                    + "] ("
                    + str(len(lst))
                    + ")"
                )
        else:
            return [goThrough(y) for y in lst]

    return goThroughDict(jsondata)


def callEtc(postdata, url, uploadfile=None):
    # Suppress SSL certificate warnings
    import warnings

    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    # If no file is to be uploaded, send a POST request with JSON data:
    if uploadfile is None:
        try:
            return requests.post(
                url,
                data=json.dumps(postdata),
                headers={"Content-Type": "application/json"},
                verify=False,
            )  # Send data with SSL verification disabled
        except Exception as e:
            print(
                "Post request without upload returned error:" + str(e)
            )  # Print error if POST fails
            sys.exit()
    else:
        # Encode postdata as URL-safe string
        encoded_data = urllib.parse.quote(json.dumps(postdata))
        # Construct the URL with filename and encoded JSON data as query parameters.
        request_url = (
            f"{url}?filename={os.path.basename(uploadfile)}&data={encoded_data}"
        )
        try:
            # Open the file to be uploaded in binary read mode
            with open(uploadfile, "rb") as f:
                # Create a dictionary for files to be uploaded
                files = {
                    "file": (os.path.basename(uploadfile), f)
                }  # Key 'file' matches the server's expected key
                # Make a POST request with the file and encoded JSON data in the URL
                return requests.post(
                    request_url, files=files, verify=False
                )  # Send file with SSL verification disabled
        except Exception as e:
            print(
                "Post request with upload returned error:" + str(e)
            )  # Print error if POST with file fails
            sys.exit()


def output(jsondata, outputfile, indent, collapse):
    if collapse:
        jsondata = collapse(jsondata)

    if outputfile:
        with open(outputfile, "w") as of:
            of.write(json.dumps(jsondata, indent=indent))
    else:
        print(json.dumps(jsondata, indent=indent))


def getEtcUrl(etcname):
    if (
        "4most" in etcname.lower()
        or "qmost" in etcname.lower()
        or "fourmost" in etcname.lower()
    ):
        return "Fourmost/"
    elif "crires" in etcname.lower():
        return "Crires2/"
    else:  # normal case
        return etcname.lower().capitalize() + "/"


def getPostdata(instrument_name, postdatafile):
    try:
        with open(postdatafile) as f:
            postdata = json.loads(f.read())
    except OSError:
        logger.error("cannot open", postdatafile)
        sys.exit()
    return postdata


def main_etc_caller(uploadfile, outputfile):
    etcname = "crires2"
    server = "https://etc.eso.org"
    baseurl = server + "/observing/etc/etcapi/"
    url = baseurl + getEtcUrl(etcname)
    indent = 4
    collapse = False

    postdata = getPostdata(etcname, uploadfile)  # prepare input
    jsondata = callEtc(postdata, url, uploadfile).json()  # output

    output(jsondata, outputfile, indent, collapse)
