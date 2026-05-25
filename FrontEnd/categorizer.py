"""
categorizer.py — Keyword-based transaction category normalization.

Categories:
  dining        restaurants, fast food, delivery, cafes
  groceries     supermarkets, food stores
  gas           fuel stations
  shopping      retail, Amazon, department stores
  subscriptions streaming, SaaS, membership fees
  bills         utilities, rent, phone, insurance
  health        pharmacy, medical, dental
  travel        flights, hotels, rideshare, transit
  essentials    household services, personal care
  miscellaneous anything that doesn't match

Rules:
  - Keyword matching is case-insensitive and uses substring search.
  - Order matters: more specific categories come before broader ones.
  - If the bank provides a raw_category, it is used as a secondary signal
    only when keyword matching yields 'miscellaneous'.
"""

from __future__ import annotations

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Keyword table  (category → tuple of lowercase substrings)
# ─────────────────────────────────────────────────────────────────────────────
_KEYWORDS: dict[str, tuple[str, ...]] = {

    "dining": (
        # Fast food
        "mcdonald", "wendy", "burger king", "taco bell", "chipotle",
        "popeyes", "chick-fil-a", "chick fil a", "kfc", "sonic drive",
        "jack in the box", "whataburger", "hardee", "carl's jr", "in-n-out",
        "five guys", "shake shack", "culver", "raising cane", "wingstop",
        "panda express", "pei wei", "qdoba",
        # Pizza
        "domino", "pizza hut", "papa john", "little caesar", "papa murphy",
        "pizza", "calzone",
        # Sub / sandwich
        "subway", "jimmy john", "jersey mike", "firehouse subs",
        "potbelly", "schlotzsky", "quiznos", "moe's sw grill",
        # Sit-down
        "applebee", "olive garden", "chili", "outback", "red lobster",
        "longhorn", "cheesecake factory", "ihop", "denny", "waffle house",
        "cracker barrel", "texas roadhouse", "buffalo wild wing",
        "red robin", "dine in",
        # Coffee
        "starbucks", "dunkin", "dutch bros", "caribou coffee",
        "coffee bean", "peets coffee", "tim horton", "panera",
        # Delivery / aggregator
        "doordash", "uber eats", "grubhub", "postmates", "seamless",
        "caviar", "eat street", "delivery.com",
        # Generic
        "restaurant", "eatery", "diner", "bistro", "brasserie",
        "bar & grill", "bar and grill", "steakhouse", "sushi",
        "noodle", "ramen", "bbq", "smokehouse", "taproom", "tavern",
        "brewery", "ice cream", "yogurt", "smoothie", "boba",
        "baskin-robbins", "dairy queen", "cold stone",
        "braum", "kalachandji", "einstein bagel", "jason's deli",
        "newk's eatery", "taziki", "zoes kitchen", "noodles & company",
    ),

    "groceries": (
        "kroger", "safeway", "albertsons", "publix", "meijer",
        "h-e-b", "heb ", "whole foods", "trader joe",
        "sprouts", "aldi", "lidl", "wegmans", "harris teeter",
        "food lion", "giant food", "stop & shop", "stop and shop",
        "price chopper", "winn-dixie", "ralphs", "vons",
        "king soopers", "smiths food", "fry's food",
        "natural grocers", "grocery outlet", "winco foods",
        "save a lot", "fiesta mart", "market basket",
        "fresh market", "lucky supermarket", "stater bros",
        "piggly wiggly", "ingles", "hannaford", "shoprite",
        "acme markets", "food 4 less", "bi-lo", "giant eagle",
        # Warehouse clubs (grocery context)
        "costco", "sam's club", "bj's wholesale",
        # Walmart in grocery context
        "walmart grocery", "walmart neighborhood", "wal-mart market",
        # Generic
        "grocery", "supermarket", "food store", "produce",
    ),

    "gas": (
        "shell", "chevron", "bp ", "exxon", "mobil", "exxonmobil",
        "sunoco", "marathon gas", "speedway", "circle k",
        "wawa", "quiktrip", "qt gas", "love's travel",
        "pilot flying j", "flying j", "casey's",
        "kwik trip", "kwik star", "kwik-e-mart",
        "ampm", "arco", "texaco", "conoco", "phillips 66",
        "gulf oil", "valero", "holiday station",
        "fuel", "gas station", "petrol",
        # Note: 7-eleven is ambiguous; prioritize gas context only
        "7-eleven gas", "7eleven gas",
    ),

    "shopping": (
        # E-commerce
        "amazon.com", "amzn.com", "amazon mktpl", "amazon prime",
        "ebay", "etsy", "shopify", "wish.com", "aliexpress",
        "wayfair", "overstock",
        # Big box retail
        "target", "walmart supercenter", "wal-mart super",
        "walmart.com", "walmart.ca",
        "best buy", "home depot", "lowe's", "lowes",
        "ikea", "menards", "ace hardware", "true value",
        # Discount / off-price
        "tj maxx", "tjmaxx", "marshalls", "ross store",
        "burlington", "tuesday morning",
        # Department stores
        "nordstrom", "macy", "bloomingdale", "jcpenney",
        "kohl", "belk", "dillard", "saks",
        # Clothing / Accessories
        "gap", "old navy", "banana republic", "forever 21",
        "h&m", "zara", "uniqlo", "lululemon", "athleta",
        "nike", "adidas", "under armour", "new balance",
        "foot locker", "footlocker", "dick's sporting", "academy sports",
        # Dollar / discount
        "dollar tree", "dollar general", "five below", "family dollar",
        "big lots",
        # Electronics
        "apple store", "apple.com", "microsoft store",
        "newegg", "b&h photo", "adorama",
        # Home / decor
        "pottery barn", "williams sonoma", "williams-sonoma",
        "crate and barrel", "crate & barrel", "west elm",
        "bed bath", "bath & body works", "bath and body",
        "pier 1", "tuesday morning", "homegoods",
        # Craft / hobby
        "hobby lobby", "michaels store", "joann fabric", "ac moore",
        # Books / misc
        "barnes & noble", "half price book",
    ),

    "subscriptions": (
        # Video streaming
        "netflix", "hulu", "disney+", "disney plus", "hbomax", "hbo max",
        "peacock", "paramount+", "paramount plus", "apple tv+", "apple tv plus",
        "amazon prime video", "crunchyroll", "funimation",
        # Music
        "spotify", "apple music", "pandora", "tidal", "amazon music",
        "youtube premium", "youtube music", "soundcloud",
        # Cloud / productivity
        "microsoft 365", "office 365", "microsoft office",
        "google one", "google workspace", "icloud",
        "adobe creative", "adobe acrobat", "adobe cc",
        "dropbox", "box.com", "notion.so",
        # Communication
        "zoom.us", "slack.com", "zoom video",
        # Podcasts / reading
        "audible", "kindle unlimited", "scribd", "lumosity",
        "patreon", "substack", "masterclass",
        # Health / wellness subscriptions
        "headspace", "calm app", "noom", "myfitnesspal",
        # Gaming
        "xbox game pass", "playstation plus", "nintendo online",
        "steam subscription", "twitch subscription",
        # Dating
        "match.com", "hinge", "bumble", "tinder gold",
        # Misc
        "xm radio", "sirius xm", "satellite radio",
        "duolingo", "babbel",
    ),

    "bills": (
        # Utilities
        "electric", "electricity", "utility", "utilities",
        "water bill", "sewage", "trash", "garbage", "waste management",
        "atmos energy", "con edison", "pg&e", "duke energy",
        "dominion energy", "xcel energy", "southern company",
        "american electric", "centerpoint energy",
        # Internet / cable / phone providers
        "comcast", "xfinity", "spectrum", "cox communicat",
        "charter communicat", "dish network", "directv",
        "at&t", "verizon wireless", "t-mobile", "sprint",
        "us cellular", "boost mobile", "cricket wireless",
        "optimum", "frontier communicat",
        # Insurance
        "progressive", "geico", "state farm", "allstate",
        "usaa", "nationwide", "liberty mutual", "farmers",
        "travelers insur", "amica", "erie insurance",
        "blue cross", "bluecross", "cigna", "aetna", "humana",
        "united health", "unitedhealthcare", "oscar health",
        "kaiser", "molina", "anthem",
        # Housing
        "rent payment", "rental payment", "mortgage", "hoa dues",
        "property tax", "home insurance",
        # Generic
        "bill pay", "autopay", "insurance premium",
    ),

    "health": (
        # Pharmacy
        "cvs pharmacy", "walgreens", "rite aid", "duane reade",
        "costco pharmacy", "walmart pharmacy", "kroger pharmacy",
        "prescription", "rx pickup",
        # Medical
        "hospital", "medical center", "urgent care",
        "doctor", "physician", "clinic", "health clinic",
        "labcorp", "quest diagnostic", "laboratory",
        "radiology", "imaging center",
        # Dental
        "dental", "dentist", "orthodont",
        # Vision
        "vision center", "lenscrafters", "pearle vision",
        "america's best", "optometry", "optometrist", "eye exam",
        # Fitness (one-time, not subscription)
        "planet fitness", "la fitness", "anytime fitness",
        "gold's gym", "24 hour fitness", "ymca",
        "equinox", "orangetheory", "crossfit",
        # Mental health
        "therapy", "counseling", "psychiatr", "psycholog",
        "betterhelp", "talkspace",
    ),

    "travel": (
        # Airlines
        "delta air", "united airline", "american airline",
        "southwest airline", "spirit airline", "frontier airline",
        "jetblue", "alaska airline", "hawaiian airline",
        "air canada", "british airway", "lufthansa",
        # Hotels
        "marriott", "hilton hotel", "hyatt", "holiday inn",
        "doubletree", "sheraton", "westin", "st. regis",
        "best western", "quality inn", "comfort inn", "hampton inn",
        "motel 6", "super 8", "days inn",
        # OTAs / booking
        "airbnb", "vrbo", "booking.com", "expedia", "hotels.com",
        "priceline", "hotwire", "kayak", "trivago",
        # Rideshare
        "uber trip", "lyft ride", "uber *trip",
        # Car rental
        "enterprise rent", "hertz", "avis rent", "budget car",
        "dollar rent", "thrifty car", "national car",
        "alamo rent", "sixt rent",
        # Transit
        "amtrak", "greyhound", "megabus", "flixbus",
        "metro card", "bus fare", "transit",
        "mta ", "bart ", "wmata", "cta fare",
        # Parking / toll
        "parking meter", "parking garage", "ez pass",
        "sunpass", "fastrak", "toll road",
        # Generic
        "travel", "airport",
    ),

    "essentials": (
        # Personal care
        "haircut", "barber", "great clips", "supercuts",
        "hair salon", "nail salon", "spa",
        # Laundry
        "laundromat", "laundry", "dry clean", "cleaners",
        # Shipping
        "usps", "fedex", "ups store", "ups shipping",
        "dhl ", "postal service",
        # Pet
        "petco", "petsmart", "pet supplies",
        "veterinary", "vet clinic", "animal hospital",
        # Baby / child
        "buy buy baby", "carter", "children's place",
        # General dollar / convenience (non-gas)
        "7-eleven", "circle k store", "casey's general",
        # Miscellaneous household
        "home repair", "plumber", "locksmith",
        "storage unit", "public storage", "extra space",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Bank raw-category fallback map
# ─────────────────────────────────────────────────────────────────────────────
_RAW_CAT_MAP: dict[str, str] = {
    # Dining
    "dining":              "dining",
    "restaurants":         "dining",
    "restaurant":          "dining",
    "fast food":           "dining",
    "food":                "dining",
    # Groceries
    "grocery":             "groceries",
    "groceries":           "groceries",
    "grocery stores":      "groceries",
    "supermarket":         "groceries",
    "supermarkets":        "groceries",
    # Gas
    "fuel":                "gas",
    "gas":                 "gas",
    "gasoline":            "gas",
    "gas stations":        "gas",
    "petroleum":           "gas",
    # Shopping / merchandise
    "merchandise":         "shopping",
    "retail":              "shopping",
    "department":          "shopping",
    "department stores":   "shopping",
    "warehouse clubs":     "shopping",
    "wholesale clubs":     "shopping",
    "home improvement":    "shopping",
    # Subscriptions
    "subscription":        "subscriptions",
    "streaming":           "subscriptions",
    "streaming services":  "subscriptions",
    "digital":             "subscriptions",
    # Bills
    "utilities":           "bills",
    "utility":             "bills",
    "telecom":             "bills",
    "insurance":           "bills",
    # Health
    "health":              "health",
    "healthcare":          "health",
    "medical":             "health",
    "pharmacy":            "health",
    "drug store":          "health",
    "drugstores":          "health",
    # Travel
    "travel":              "travel",
    "travel/entertainment":"travel",
    "transportation":      "travel",
    "airline":             "travel",
    "hotel":               "travel",
    "rideshare":           "travel",
    # Essentials
    "personal care":       "essentials",
    "services":            "essentials",
    "automotive":          "essentials",
    "education":           "essentials",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def categorize(merchant_raw: str, raw_category: Optional[str] = None) -> str:
    """
    Return the normalized spending category for a transaction.

    1. Try keyword matching on the raw merchant string.
    2. If that yields 'miscellaneous', try interpreting the bank's raw category.
    3. Fall back to 'miscellaneous'.
    """
    if not merchant_raw:
        return _from_raw_cat(raw_category)

    needle = merchant_raw.lower()

    for category, keywords in _KEYWORDS.items():
        for kw in keywords:
            if kw in needle:
                return category

    # Keyword matching gave nothing; try the bank's own label
    return _from_raw_cat(raw_category)


def _from_raw_cat(raw_category: Optional[str]) -> str:
    if not raw_category:
        return "miscellaneous"
    needle = raw_category.lower().strip()
    for key, cat in _RAW_CAT_MAP.items():
        if key in needle:
            return cat
    return "miscellaneous"
