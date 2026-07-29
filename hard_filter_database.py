import json
import os

from cache_query import CacheQuery
from query_understanding import QueryParser, MODEL
from read_questions import load_questions
# Assuming your Pydantic models (HardFilters, QuerySpec) are imported here
from data_used_by_llm.schemas import HardFilters, BusinessModel


# given a certain company (from the jsonl) and some filters for a query
# compute the score as a percentage
def score_company(company: dict, filters: HardFilters, business_models: BusinessModel) -> tuple[float, int]:
    active_conditions = 0
    met_conditions = 0

    eliminate = False  # Fixed from sympy 'false'

    # make a filter list for countries and regions
    filter_geos = set()
    if filters.countries:
        filter_geos.update(str(c).lower() for c in filters.countries)
    if filters.regions:
        filter_geos.update(str(r).lower() for r in filters.regions)

    # Only process geography if the query actually asked for it
    if filter_geos:
        # Active conditions = number of unique locations requested
        active_conditions += len(filter_geos)

        # 2. Build the company list (lowercase)
        company_geos = set()
        address = company.get("address", {})

        # Add country (handling both string or unexpected list formats)
        comp_country = address.get("country")
        if comp_country:
            if isinstance(comp_country, list):
                company_geos.update(str(c).lower() for c in comp_country)
            else:
                company_geos.add(str(comp_country).lower())

        # Add region (handling both string or unexpected list formats)
        comp_region = address.get("REGION_BIG")
        if comp_region:
            if isinstance(comp_region, list):
                company_geos.update(str(r).lower() for r in comp_region)
            else:
                company_geos.add(str(comp_region).lower())

        # In case list for company is empty, then jump over
        if company_geos:
            # Find the overlap between requested locations and company locations
            intersections = filter_geos.intersection(company_geos)

            if intersections:
                # Add the number of matching elements
                met_conditions += len(intersections)
            else:
                # Mark for elimination if they have locations, but NONE match
                eliminate = True


    # Minimum Employees
    if filters.min_employees is not None:
        active_conditions += 1
        emp = company.get("employee_count")
        if emp is not None and emp >= filters.min_employees:
            met_conditions += 1
        elif emp is not None:
            eliminate = True  # Mark for elimination if employee count doesn't match

    # Maximum Employees
    if filters.max_employees is not None:
        active_conditions += 1
        emp = company.get("employee_count")
        if emp is not None and emp <= filters.max_employees:
            met_conditions += 1
        elif emp is not None:
            eliminate = True  # Mark for elimination if employee count doesn't match

    # Minimum Revenue
    if filters.min_revenue is not None:
        active_conditions += 1
        rev = company.get("revenue")
        if rev is not None and rev >= filters.min_revenue:
            met_conditions += 1
        elif rev is not None:
            eliminate = True  # Mark for elimination if revenue doesn't match

    # Maximum Revenue
    if filters.max_revenue is not None:
        active_conditions += 1
        rev = company.get("revenue")
        if rev is not None and rev <= filters.max_revenue:
            met_conditions += 1
        elif rev is not None:
            eliminate = True  # Mark for elimination if revenue doesn't match

    # Founded After
    if filters.founded_after is not None:
        active_conditions += 1
        year = company.get("year_founded")
        if year is not None and year >= filters.founded_after:
            met_conditions += 1
        elif year is not None:
            eliminate = True  # Mark for elimination if founded year doesn't match

    # Founded Before
    if filters.founded_before is not None:
        active_conditions += 1
        year = company.get("year_founded")
        if year is not None and year <= filters.founded_before:
            met_conditions += 1
        elif year is not None:
            eliminate = True  # Mark for elimination if founded year doesn't match

    # Public Status
    if filters.is_public is not None:
        active_conditions += 1
        if company.get("is_public") == filters.is_public:
            met_conditions += 1
        elif company.get("is_public") is not None:
            eliminate = True  # Mark for elimination if public status doesn't match

    # if business_models:required is not present inside the company business models
    if business_models:
        company_business_models = company.get("business_model", [])
        # print(f"Company business models: {company_business_models}, Required business model: {business_models}")
        if business_models and not any(bm in company_business_models for bm in business_models):
            eliminate = True
        elif business_models is not None:
            met_conditions += 10

    # if even a single one is not met, we eliminate the company from the results
    if eliminate:
        return 0, active_conditions

    # Calculate Score
    if active_conditions == 0:
        # in case there were no conditions from query, pass it directly
        return 1, active_conditions

    return met_conditions / active_conditions, active_conditions


def generate_filtered_subset(
        source_file: str,
        dest_file: str,
        query_spec
):
    """
    Reads the enhanced companies file, scores them based on QuerySpec,
    and writes passing companies to a temporary JSONL file.
    """
    if not os.path.exists(source_file):
        print(f"Error: Could not find {source_file}")
        return None

    passed_count = 0
    filters = query_spec.hard_filters
    required_business_models = query_spec.business_models.required

    companies = []

    with open(source_file, 'r', encoding='utf-8') as infile, \
            open(dest_file, 'w', encoding='utf-8') as outfile:


        for line in infile:
            if not line.strip():
                continue

            company = json.loads(line)

            # 1. Score the company
            score, active_conditions = score_company(company, filters, required_business_models)

            # If the score is higher than 0 (it has no missed requirements).
            if score > 0:
                # inject the score for latter LLM quering
                company["_filter_score"] = score
                company["_nr_of_existing_filters"] = active_conditions

                # write to the tmp file
                outfile.write(json.dumps(company) + '\n')
                companies.append(company)
                passed_count += 1

    return passed_count, companies


if __name__ == "__main__":
    cache = CacheQuery()
    parser = QueryParser(model=MODEL, querycache=cache)

    SOURCE_DB = "data/companies_enhanced.jsonl"

    for item in load_questions():
        number, question = item["number"], item["question"]

        # Parse the question via LLM / logic
        spec = parser.parse(question)
        print(f"\n=== {number}. {question}")

        # Define a unique tmp file for this query's results
        tmp_output_file = f"data/tmp_query_{number}_filtered.jsonl"

        survivors, _ = generate_filtered_subset(
            source_file=SOURCE_DB,
            dest_file=tmp_output_file,
            query_spec=spec
        )

        print(f"[{spec.complexity.upper()}] Filtered down to {survivors} candidates. Saved to {tmp_output_file}")