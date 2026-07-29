# prompts.py
from data_used_by_llm.vocab import BM_PROMPT_LIST

SYSTEM_PROMPT = f"""You parse company-search queries into structured JSON specs.

hard_filters — only constraints EXPLICITLY stated in the query. Never invent
thresholds that are not there. Specific countries ("Germany", "France") go in
countries; broader areas ("Europe", "Scandinavia") go in regions.

business_models — how the target company operates. Valid values:
{BM_PROMPT_LIST}.
- required: business-STRUCTURE terms stated in the query.
  B2B -> Business-to-Business, B2C -> Business-to-Consumer,
  SaaS -> Software-as-a-Service, e-commerce -> E-commerce,
  wholesaler/distributor -> Wholesale.
- preferred: OPERATIONAL hints, stated or implied.
  manufacturer/producer -> Manufacturing, supplier -> Wholesale,
  logistics/freight -> Logistics/Transportation, retailer -> Retail,
  SaaS additionally implies Subscription-Based.
- Leave both empty if the query says nothing about business model.
- Watch the direction: in "supplies X to D2C brands", D2C describes the
  CUSTOMER. Never assign the customer's model to the target company.
- Industries with no matching value (fintech, HR, pharma, construction)
  belong in key_terms, not here.

ideal_match_description — 2-3 sentences profiling the perfect matching
COMPANY as if writing its profile: what it makes or does, and who it sells
to. Name concrete products or services. Describe the supplier side of any
relationship, never the customer side.

key_terms — 1-4 terms. weight 1.0 for what defines the company itself,
0.3-0.5 for context such as the customer's industry.

complexity — exactly one of:
- "structured": every constraint maps to a data field (location, size,
  revenue, public status, founding year, plain industry name)
- "hybrid": field filters plus a semantic constraint
- "judgment": requires reasoning about roles, supply chains, competition,
  or vague qualifiers ("fast-growing", "critical", "could supply")

reasoning — one sentence justifying the complexity label."""


FEW_SHOT = [
    {"role": "user", "content": "Query: Automotive parts manufacturers in Japan"},
    {"role": "assistant", "content": """{
  "hard_filters": {"countries": ["Japan"], "regions": [], "min_employees": null,
    "max_employees": null, "min_revenue": null, "max_revenue": null,
    "founded_after": null, "founded_before": null, "is_public": null},
  "business_models": {"required": [], "preferred": ["Manufacturing"]},
  "ideal_match_description": "Manufactures automotive components such as engine parts, braking systems, sensors and interior modules. Sells B2B to car makers and tier-1 suppliers.",
  "key_terms": [{"term": "automotive parts", "weight": 1.0},
                {"term": "manufacturer", "weight": 0.7}],
  "complexity": "structured",
  "reasoning": "Location and a plain industry name map directly to data fields."
}"""},

    {"role": "user", "content": "Query: Companies that could supply fabrics to fashion brands"},
    {"role": "assistant", "content": """{
  "hard_filters": {"countries": [], "regions": [], "min_employees": null,
    "max_employees": null, "min_revenue": null, "max_revenue": null,
    "founded_after": null, "founded_before": null, "is_public": null},
  "business_models": {"required": [], "preferred": ["Manufacturing", "Wholesale"]},
  "ideal_match_description": "Produces or distributes textiles: woven and knitted fabrics, denim, leather and technical fibers. Sells B2B to apparel and fashion companies.",
  "key_terms": [{"term": "fabric", "weight": 1.0},
                {"term": "textile", "weight": 0.9},
                {"term": "fashion", "weight": 0.4}],
  "complexity": "judgment",
  "reasoning": "'Could supply' requires reasoning about a supply-chain role; the target is the supplier, not the fashion brand."
}"""},

    {"role": "user", "content": "Query: B2B SaaS platforms for accounting teams in Europe"},
    {"role": "assistant", "content": """{
  "hard_filters": {"countries": [], "regions": ["Europe"], "min_employees": null,
    "max_employees": null, "min_revenue": null, "max_revenue": null,
    "founded_after": null, "founded_before": null, "is_public": null},
  "business_models": {"required": ["Business-to-Business", "Software-as-a-Service"],
                      "preferred": ["Subscription-Based"]},
  "ideal_match_description": "Develops cloud software for accounting and finance teams: bookkeeping, invoicing, expense management and financial reporting. Sold to businesses on a subscription basis.",
  "key_terms": [{"term": "accounting software", "weight": 1.0},
                {"term": "finance", "weight": 0.5}],
  "complexity": "hybrid",
  "reasoning": "Region and business model are field filters; 'accounting solutions' is a semantic constraint."
}"""},
]