from enum import Enum

class BusinessModel(str, Enum):
    ASSET_MANAGEMENT = "Asset Management"
    B2B = "Business-to-Business"
    B2C = "Business-to-Consumer"
    ECOMMERCE = "E-commerce"
    ENTERPRISE = "Enterprise"
    ENTERTAINMENT = "Entertainment"
    FRANCHISING = "Franchising"
    GOVERNMENT = "Government/Public Sector"
    LICENSING = "Licensing"
    LOGISTICS = "Logistics/Transportation"
    MANUFACTURING = "Manufacturing"
    NONPROFIT = "Non-Profit/Non-Governmental Organization"
    REAL_ESTATE = "Real Estate Development"
    RETAIL = "Retail"
    SERVICE_PROVIDER = "Service Provider"
    SAAS = "Software-as-a-Service"
    SUBSCRIPTION = "Subscription-Based"
    WHOLESALE = "Wholesale"

BM_PROMPT_LIST = ", ".join(m.value for m in BusinessModel)

# (regex on lowercased query, value, tier)
BM_RULES: list[tuple[str, BusinessModel, str]] = [
    (r"\bb2b\b|business.to.business",    BusinessModel.B2B,           "required"),
    (r"\bsaas\b|software.as.a.service",  BusinessModel.SAAS,          "required"),
    (r"\be.?commerce\b",                 BusinessModel.ECOMMERCE,     "required"),
    (r"\bwholesal",                      BusinessModel.WHOLESALE,     "required"),
    (r"manufactur",                      BusinessModel.MANUFACTURING, "preferred"),
    (r"\bsuppl(y|ier|ies|ying)\b",       BusinessModel.WHOLESALE,     "preferred"),
    (r"\blogistics?\b|freight",          BusinessModel.LOGISTICS,     "preferred"),
    (r"subscription",                    BusinessModel.SUBSCRIPTION,  "preferred"),
    (r"\bretail(er)?\b",                 BusinessModel.RETAIL,        "preferred"),
]