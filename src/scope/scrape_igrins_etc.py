"""
pings the IGRINS exposure time calculator.
"""
import re

from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver.common.by import By

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

    driver = webdriver.Chrome()
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
