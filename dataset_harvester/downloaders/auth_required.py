"""
Auth-required repository instructions.

These repositories need credentials or special API clients.
Instead of failing silently, we print clear setup instructions.
"""

_INSTRUCTIONS = {
    "nsidc": """
  NSIDC / NASA Earthdata — requires free account
  1. Register at: https://urs.earthdata.nasa.gov/users/new
  2. Install earthaccess:  pip install earthaccess
  3. Then run:
       import earthaccess
       earthaccess.login()
       earthaccess.download(results, local_path="downloads/nsidc/")
""",
    "nasa_earthdata": """
  NASA Earthdata — requires free account
  1. Register at: https://urs.earthdata.nasa.gov/users/new
  2. Install earthaccess:  pip install earthaccess
  3. Search and download via earthaccess or NASA Earthdata Search:
       https://search.earthdata.nasa.gov/
""",
    "copernicus_cds": """
  Copernicus Climate Data Store (ERA5, etc.) — requires free account + API key
  1. Register at: https://cds.climate.copernicus.eu/
  2. Get your API key from: https://cds.climate.copernicus.eu/profile
  3. Install cdsapi:  pip install cdsapi
  4. Create ~/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api
       key: YOUR_API_KEY
  5. Then use cdsapi to download — see dataset page for the request dict.
  CDS API key detected in .env: YES (key already present)
""",
    "copernicus_marine": """
  Copernicus Marine Service (TOPAZ4, etc.) — requires free account
  1. Register at: https://marine.copernicus.eu/
  2. Install copernicus-marine-client:  pip install copernicusmarine
  3. Then run:
       import copernicusmarine
       copernicusmarine.login()
       copernicusmarine.get(dataset_id="ARCTIC_ANALYSISFORECAST_PHY_002_001",
                            output_directory="downloads/topaz4/")
""",
    "ecmwf": """
  ECMWF (ERA-Interim archive) — requires free account + ECMWF API key
  Note: ERA-Interim was discontinued in 2019. Archive still accessible.
  1. Register at: https://www.ecmwf.int/user/login
  2. Install ecmwf-api-client:  pip install ecmwf-api-client
  3. Get your API key from: https://api.ecmwf.int/v1/key/
  4. Create ~/.ecmwfapirc with:
       {"url": "https://api.ecmwf.int/v1", "key": "YOUR_KEY", "email": "YOUR_EMAIL"}
""",
    "esa": """
  ESA (CryoSat-2, Sentinel data) — requires free ESA Earth Online account
  1. Register at: https://earth.esa.int/eogateway/
  2. For CryoSat-2: https://earth.esa.int/eogateway/missions/cryosat/data
  3. For Sentinel: use Copernicus Data Space (no auth needed for Sentinel):
       https://browser.dataspace.copernicus.eu/
""",
}


def print_instructions(repository: str, dataset_name: str):
    instructions = _INSTRUCTIONS.get(repository)
    if instructions:
        print(f"\n  [AUTH REQUIRED] {dataset_name}")
        print(f"  Repository: {repository}")
        print(instructions)
    else:
        print(f"\n  [AUTH REQUIRED] {dataset_name} ({repository})")
        print(f"  Visit the repository landing page to download manually.")
