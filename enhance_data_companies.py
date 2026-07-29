import pandas as pd
import json
import os
import ast


# helper for loading json file
def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# map the country code to country if non-existent
COUNTRY_CODE_MAP: dict[str, str] = {}


# read CountryCodesRaw and return a dictionary for code and country
def load_country_codes():
    countries_data = load_json("data/CountryCodesRaw.json")
    # Create the dictionary and convert the code to lowercase
    COUNTRY_CODE_MAP.update({country["code"].lower(): country["name"] for country in countries_data})


# Map the region to the country. (Consider moving more specific regions to the top)
REGION_MAP: dict[str, list[str]] = {
    "scandinavia": ["Sweden", "Norway", "Denmark"],
    "nordics": ["Finland", "Iceland", "Sweden", "Norway", "Denmark"],
    "dach": ["Germany", "Austria", "Switzerland"],
    "benelux": ["Belgium", "Netherlands", "Luxembourg"],
    "europe": [
        "France", "United Kingdom", "Italy", "Spain",
        "Poland", "Romania", "Czech Republic", "Portugal", "Ireland",
        "Greece", "Hungary", "Bulgaria", "Croatia", "Slovakia", "Slovenia",
        "Lithuania", "Latvia", "Estonia",
        "Germany", "Austria", "Switzerland",
        "Belgium", "Netherlands", "Luxembourg",
        "Sweden", "Norway", "Denmark", "Finland", "Iceland",
    ],
    "north america": ["United States", "Canada", "Mexico"],
}

# replace ' "{ ' and ' }" ' with '{' and '}'
def eliminate_useless_strings(address_raw):
    # only try to replace things if it's actually a string
    if isinstance(address_raw, str):
        return (address_raw
                .replace('"{', '{')
                .replace('}"', '}'))

    return address_raw

import ast

# turn a stringified dictionary back into a dictionary
def parse_stringified_dict(cell_value):
    if isinstance(cell_value, str):
        try:
            return ast.literal_eval(cell_value)
        except (SyntaxError, ValueError):
            return cell_value
    return cell_value

def enhance_address(address_raw):
    # apply country and region logic to an address (region being smth like scandinavia or europe)
    # skip this company entirely if there is no address block (NaN or empty)
    if pd.isna(address_raw) or not address_raw:
        return address_raw

    # Convert stringified dictionary if needed (safeguard based on your data)
    if isinstance(address_raw, str):
        try:
            address = ast.literal_eval(address_raw)
        except (SyntaxError, ValueError):
            return address_raw
    else:
        # Create a copy so we don't mutate the original reference directly
        address = address_raw.copy() if isinstance(address_raw, dict) else address_raw

    if not isinstance(address, dict):
        return address

    # fill missing country names using the code
    # .get() returns "" if "country" doesn't exist, preventing KeyErrors
    if address.get("country", "") == "":
        country_code = address.get("country_code", "").lower()
        if country_code in COUNTRY_CODE_MAP:
            address["country"] = COUNTRY_CODE_MAP[country_code]

    # add REGION_BIG based on the country
    # .get() again protects us from KeyErrors
    country_name = address.get("country")
    if country_name:
        regions = []
        for region, countries in REGION_MAP.items():
            if country_name in countries:
                regions.append(region)
        if regions:
            address["REGION_BIG"] = regions

    return address


def sync_and_modify(source_file, dest_file):
    # current companies.json file
    df = pd.read_json(source_file, lines=True)



    if 'address' in df.columns:
        # fix the bug where the address is as a big text instead of a dictionary
        df['address'] = df['address'].apply(eliminate_useless_strings)
        # why tf is this bug also in primary NAICS. These mf did this to catch full ai solutions
        df['primary_naics'] = df['primary_naics'].apply(eliminate_useless_strings)
        df['primary_naics'] = df['primary_naics'].apply(enhance_address)
        # Apply our modification logic specifically to the address column
        df['address'] = df['address'].apply(enhance_address)

    # Save the dataframe back to a JSON Lines file
    df.to_json(dest_file, orient='records', lines=True, force_ascii=False)


if __name__ == "__main__":
    # Updated to match your path suggestion
    SOURCE = 'data/companies.jsonl'
    DEST = 'data/companies_enhanced.jsonl'

    # initialize your map FIRST!
    load_country_codes()

    # run the synchronizer
    sync_and_modify(SOURCE, DEST)
    print("Enhancement complete!")